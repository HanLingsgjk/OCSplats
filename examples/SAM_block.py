#这个项目是用SAM把要读取的场景，打成碎块。
#并且特征提取Sam 和 Dinov2
#运行前置条件，首先需要重建完成
import cv2
import sys
import torch
import os
import time
from PIL import Image
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
from glob import glob
import os.path as osp
import matplotlib.pyplot as plt
from ultralytics import YOLO
modeldect = YOLO("/home/lh/CSCV_occ/yolov8x.pt")
from segment_anything import SamAutomaticMaskGenerator, sam_model_registry, SamPredictor
print('Load anything model...')
sam = sam_model_registry['vit_h'](checkpoint='/home/lh/Track-Anything/checkpoints/sam_vit_h_4b8939.pth')
_ = sam.cuda()
print('Load dinov2 model...')
device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
dinov2_vits14.to(device)
extractor = dinov2_vits14



ifshow=0
import pycolmap.pycolmap as pycolmap
#搞一个版本，只找正面区域，然后最细化分块

# 更新内容，只考虑边缘区域附近的
def get_wlmask(rgb_pred,scale):

    rgb_pred_gray = cv2.cvtColor(rgb_pred, cv2.COLOR_RGB2GRAY)

    sobelx = cv2.Sobel(rgb_pred_gray / 255.0, cv2.CV_64F, 1, 0, ksize=7)
    sobely = cv2.Sobel(rgb_pred_gray / 255.0, cv2.CV_64F, 0, 1, ksize=7)
    # 只统计梯度明显区域的方向
    val = np.sqrt(sobelx * sobelx + sobely * sobely)
    masku = (val > 10).astype(np.float64)
    maskus = cv2.blur(masku, (int(20*scale), int(20*scale)))

    return maskus

#置信度，膜序号掩膜，黑名单掩膜
def writeConf(filename, Conf, mask_conf,not_use):
    Conf = Conf.clip(0,32)
    Conf = Conf[:, :, np.newaxis] * 2048  #0-32
    not_use = not_use[:, :, np.newaxis]  # 0-32
    mask_conf = mask_conf[:, :, np.newaxis]  # 值域为0-8
    all = np.concatenate([Conf, mask_conf,not_use], axis=-1).astype(np.uint16)
    cv2.imwrite(filename, all)

def bilinear_sampler(img, coords, mode='bilinear'):
    """ Wrapper for grid_sample, uses pixel coordinates """
    coords = torch.from_numpy(coords)
    img = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)

    coords = coords.unsqueeze(0).unsqueeze(0).float()
    img = img.float()
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1, 1], dim=-1)
    xgrid = 2 * xgrid / (W - 1) - 1
    ygrid = 2 * ygrid / (H - 1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True, mode=mode)


    return img.numpy()[0,0,0]

# 这个要搞一个自己的姿态加载
class SceneManager(pycolmap.SceneManager):
    """COLMAP pose loader.

  Minor NeRF-specific extension to the third_party Python COLMAP loader:
  google3/third_party/py/pycolmap/scene_manager.py
  """
    def intersection(self, m, n):  # 计算两个图片m,n间的交集，输出交集和交点数量
        list1 = self.images[m].point3D_ids
        list2 = self.images[n].point3D_ids
        set1 = set(list1)
        set2 = set(list2)
        common_elements = set1.intersection(set2)
        common_elements = np.array(list(common_elements))
        common_elements = np.sort(common_elements)
        return common_elements[1:], common_elements.shape[0] - 1

    def process(self, losspp=False, losscd=False):

        self.load_cameras()
        self.load_images()
        self.load_points3D()
        #首先出现次数和质量过少的SFM点剔除
        #首先剔除误差很大的点，然后出现次数过少的点
        good_mask =  np.zeros_like(self.point3D_errors)
        invid =   dict(map(reversed,self.point3D_id_to_point3D_idx.items()))
        for i in range(self.point3D_errors.size):
            Fr = float(self.point3D_id_to_images[invid[i]].shape[0])/float(self.images.__len__())
            good_mask[i] = Fr
        return self.images, self.point3D_errors,self.point3D_ids,good_mask

# 这个是everything模式
generator = SamAutomaticMaskGenerator(sam, output_mode="binary_mask", points_per_side=32,
                                      stability_score_thresh=0.92,
                                      crop_n_layers=1,
                                      crop_n_points_downscale_factor=2,
                                      min_mask_region_area=100, )
predictor = SamPredictor(sam)
#后续改进的时候可以考虑引入DAM的信息
def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:, :, 3] = 0
    for ann in sorted_anns:
        if ann['area'] > 600:
            m = ann['segmentation']
            color_mask = np.concatenate([np.random.random(3), [0.35]])
            img[m] = color_mask
    ax.imshow(img)

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def on_the_way(path, Scale):
    colmap_dir = os.path.join(path, "sparse/0/")
    cam_extrinsics, point3d_errors, point3d_ids, good_mask = SceneManager(colmap_dir).process()
    if Scale != 1:
        impathr = 'images_' + str(Scale) + '/'
    else:
        impathr = 'images/'
    impathfeature = 'images/'
    root = path + impathr
    rootf = path + impathfeature
    output_filename_SFM = os.path.join(path, 'SFM_maskV6/')
    if os.path.exists(output_filename_SFM) == False:
        os.makedirs(output_filename_SFM)
    output_filename_viz = os.path.join(path, 'Conf_vizV6/')
    if os.path.exists(output_filename_viz) == False:
        os.makedirs(output_filename_viz)
    image1o = sorted(glob(osp.join(root, '*')))
    image1f = sorted(glob(osp.join(rootf, '*')))
    with torch.no_grad():
        for j in range(image1o.__len__()):
            i = j
            imname = cam_extrinsics[i + 1].name
            # imname = imname.split('.')[0]+'.jpg'
            impath = root + imname
            impathf = rootf + imname
            img1 = cv2.imread(impath)
            img1f = cv2.imread(impathf)
            imgp = Image.open(impathf).convert('RGB')
            fileid = imname.split('.')[0] + '.png'
            print(fileid)

            time1 = time.time()
            print('get_samfeature...')
            predictor.set_image(img1f)
            features = predictor.features
            img_type = impathf[-4:]
            save_path = impathf.replace(f'{img_type}', '.npy').replace('/images/', f'/features_sam/')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.save(save_path, features.detach().cpu().numpy())
            print('get_dinofeature...')

            IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
            IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)
            RESIZE_H = (imgp.size[1] // 4) // 14 * 14
            RESIZE_W = (imgp.size[0] // 4) // 14 * 14
            transform = T.Compose([
                T.Resize((RESIZE_H, RESIZE_W)),
                T.ToTensor(),
                T.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
            ])
            img = transform(imgp)[:3].unsqueeze(0)
            features_dict = extractor.forward_features(img.cuda())
            features = features_dict['x_norm_patchtokens'].view(RESIZE_H // 14, RESIZE_W // 14, -1)
            save_path = impathf.replace(f'{img_type}', '.npy').replace('/images/', f'/Dino_v2/')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.save(save_path, features.detach().cpu().numpy())
            print('get_sam_block...')
            time2 = time.time()
            print('timeuse:', time2 - time1)

            # Anything版本的分割
            anything_masks = generator.generate(img1)
            image = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            # 计算全图尺度系数
            ws, hs, _ = image.shape
            confws = (ws / 1000) * (hs / 1000)
            wlmask = get_wlmask(img1, confws) > 0.1
            # TODO !!!!!!!!!!!!!!!!!尺度缩放
            cam_extrinsics[i + 1].points2D = cam_extrinsics[i + 1].points2D * (1 / Scale)
            pointerror = np.zeros_like(cam_extrinsics[i + 1].point3D_ids).astype(np.float32)
            for k in range(pointerror.size):
                if cam_extrinsics[i + 1].point3D_ids[k] >= 0:
                    idi = np.where(point3d_ids == cam_extrinsics[i + 1].point3D_ids[k])
                    if good_mask[idi] > 0.05:
                        pointerror[k] = point3d_errors[idi]
            used_mask = (pointerror > 0)

            if ifshow:
                plt.imshow(image)
                show_anns(anything_masks)
                plt.axis('off')
                plt.show()
                plt.imshow(image)
                plt.scatter(cam_extrinsics[i + 1].points2D[:, 0], cam_extrinsics[i + 1].points2D[:, 1], s=1, c='r')
                plt.show()
                plt.imshow(image)
                plt.scatter(cam_extrinsics[i + 1].points2D[used_mask, 0], cam_extrinsics[i + 1].points2D[used_mask, 1],
                            s=1, c='r')
                plt.show()

            # 目标级别的分割
            bg_mask = torch.ones_like(torch.from_numpy(img1[:, :, 0])).bool()
            fg_masks = torch.zeros_like(torch.from_numpy(img1[:, :, 0])).bool()
            obj_mask = torch.zeros_like(torch.from_numpy(img1[:, :, 0])).int()
            conf_mask = torch.zeros_like(torch.from_numpy(img1[:, :, 0])).float()
            not_use = torch.zeros_like(torch.from_numpy(img1[:, :, 0])).float()
            objunm = 1
            results = modeldect.predict(source=img1, show=False)
            predictor.set_image(img1)
            input_conf = results[0].boxes.conf
            input_boxes = results[0].boxes.xyxy[input_conf > 0.4]
            input_cls = results[0].boxes.cls[input_conf > 0.4]

            transformed_boxes = predictor.transform.apply_boxes_torch(input_boxes, image.shape[:2])

            sorted_masks = sorted(anything_masks, key=(lambda x: x['area']), reverse=True)
            for bg_m in sorted_masks:
                segm = bg_m['segmentation'].astype(np.uint8)
                # TODO 这里计算连通域，独立计算每个连通域，太小的丢掉
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(segm, connectivity=8)
                for n in range(1, num_labels):
                    x0, y0, w, h, numa = stats[n]
                    if numa > 1666:  # 只要是足够大小的块都堆进去
                        segmu = labels == (n)
                        iob = segmu[bg_mask].sum() / segmu.sum()
                        obj_mask[segmu] = objunm
                        objunm = objunm + 1
                        print(iob)
            # 后叠加前景
            if input_cls.shape[0] > 0:
                masks, _, _ = predictor.predict_torch(
                    point_coords=None,
                    point_labels=None,
                    boxes=transformed_boxes,
                    multimask_output=False,
                )
                for b in range(masks.shape[0]):
                    if input_cls[b] not in [77, 73, 55, 54, 53, 52, 79]:
                        fg_mask = masks[b, 0]
                        bg_mask[fg_mask] = 0
                        fg_masks[fg_mask] = 1
                        obj_mask[fg_mask] = objunm
                        objunm = objunm + 1
            if ifshow:
                plt.imshow(bg_mask)
                plt.show()
                plt.imshow(fg_masks)
                plt.show()

                plt.imshow(image)

                for mask in masks:
                    show_mask(mask.cpu().numpy(), plt.gca(), random_color=True)
                for box in input_boxes:
                    show_box(box.cpu().numpy(), plt.gca())
                plt.axis('off')
                plt.show()

                plt.imshow(obj_mask)
                plt.show()

                obj_mask[obj_mask == 0] = 255
                plt.imshow(obj_mask)
                plt.show()
                print('ok')

            obj_masks = obj_mask.numpy()
            # 现在开始统计正向引导
            numnu = 1
            for bj in range(objunm - 1):
                masksobj = obj_masks == (bj + 1)

                umask = bilinear_sampler(masksobj, cam_extrinsics[i + 1].points2D[used_mask], 'nearest')
                eu = pointerror[used_mask][umask > 0]
                ou = eu.sum()
                olden = conf_mask[masksobj].mean()
                # 下面开始计算抑制反向引导区域,这个可以顺便把潜在的反向抑制区域划出来
                if ou > 0:
                    conf = 1 / eu
                    good_en = (conf.sum() * 1280 * confws) / masksobj[wlmask].sum()
                    if good_en > 0.1 and good_en > olden:
                        conf_mask[masksobj] = good_en
            conf_viz = (conf_mask.numpy() * 100).clip(0, 255).astype(np.uint8)
            print('Conf_end')
            writeConf(os.path.join(output_filename_SFM, fileid), conf_mask.numpy(), obj_masks, not_use.numpy())
            cv2.imwrite(os.path.join(output_filename_viz, fileid), conf_viz)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--seq', type=str)
    args = parser.parse_args()
    path = '/home/lh/all_datasets/RobustScene/testsocsplat/corner/'
    scale = 4#rate[idi]
    on_the_way(path,scale)