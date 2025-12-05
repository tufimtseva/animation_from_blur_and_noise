import os
import pickle
import yaml
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
from os.path import join, exists, basename
import warnings
import cv2 as cv

try:
    import augment as A
    from flow_viz import trend_plus_vis, flow_to_image
except:
    import data.augment as A
    from data.flow_viz import trend_plus_vis, flow_to_image

warnings.filterwarnings("ignore")


class BAistPP(Dataset):
    """
    Dataset class of b-aist++ videos, each video must be represented as a dir
    Structure under each video dir:
        blur: {:08d}.png (start from  00000000.png)
        blur_anno: {:08d}.pkl (start from 00000000.pkl)
        sharp: {:08d}_{:03d}.png (start from 00000000_000.png)
        sharp_anno: {:08d}_{:03d}.pkl (start from 00000000_000.pkl)
        trend: {:08d}_trend.npy (start from 00000000_trend.npy)
    """
    def __init__(self, set_type, root_dir, suffix, num_gts, num_fut, num_past,
                 video_list=None, aug_args=None, use_trend=False,
                 temporal_step=1,
                 use_flow=False, noisy=False, noise_level=10, **kwargs):

        self.use_trend = False
        self.use_flow = False
        self.noisy = noisy
        self.sigma = noise_level

        self.img_transform, self.vid_transform = self.gen_transform(
            aug_args[set_type]
        )

        assert isinstance(root_dir, list)
        self.samples = []

        # Define train and test video lists (from your actual GoPro folder)
        train_videos = [
            "GOPR0372_07_00", "GOPR0372_07_01", "GOPR0374_11_00",
            "GOPR0374_11_01", "GOPR0374_11_02", "GOPR0374_11_03",
            "GOPR0378_13_00", "GOPR0379_11_00", "GOPR0380_11_00",
            "GOPR0384_11_01", "GOPR0384_11_02", "GOPR0384_11_03",
            "GOPR0384_11_04", "GOPR0385_11_00", "GOPR0386_11_00",
            "GOPR0477_11_00", "GOPR0857_11_00", "GOPR0868_11_01",
            "GOPR0868_11_02", "GOPR0871_11_01", "GOPR0881_11_00",
            "GOPR0884_11_00"
        ]

        test_videos = [
            "GOPR0384_11_00", "GOPR0384_11_05", "GOPR0385_11_01",
            "GOPR0396_11_00", "GOPR0410_11_00", "GOPR0854_11_00",
            "GOPR0862_11_00", "GOPR0868_11_00", "GOPR0869_11_00",
            "GOPR0871_11_00", "GOPR0881_11_01"
        ]

        # Choose split based on set_type
        if set_type == 'train':
            split_folder = 'train'
            selected_videos = train_videos
        else:  # 'valid' or 'test'
            split_folder = 'test'
            selected_videos = test_videos

        print(f"[INFO] Loading {set_type} data from {split_folder}/ folder")

        for rdir in root_dir:
            if rdir.endswith('/'):
                rdir = rdir[:-1]

            data_root = os.path.join(rdir, split_folder)

            # Auto-detect videos if folder exists
            if os.path.exists(data_root):
                available_videos = [d for d in os.listdir(data_root)
                                    if os.path.isdir(os.path.join(data_root, d))]
                # Use intersection of selected and available
                videos_to_use = [v for v in selected_videos if v in available_videos]

                # If no match, use all available videos
                if not videos_to_use:
                    videos_to_use = available_videos
                    print(f"[WARN] No matching videos found, using all {len(videos_to_use)} available")
            else:
                print(f"[ERROR] Path does not exist: {data_root}")
                continue

            try:
                new_samples = self.gen_samples_gopro(
                    data_root,
                    videos_to_use,
                    suffix,
                    num_gts,
                    num_fut,
                    num_past
                )
                self.samples += new_samples
            except Exception as e:
                print(f"[ERROR] {e}")
                import traceback
                traceback.print_exc()

        self.samples = self.samples[::temporal_step]
        print(f"[INFO] Loaded {len(self.samples)} {set_type} samples")

    def gen_transform(self, aug_args):
        """
        Generate replay transform based on augmentation arguments in the config file
        """
        img_transform, vid_trasnform = [], []
        for key, val in aug_args['image'].items():
            img_transform.append(getattr(A, key.split('_')[0])(**val))
        for key, val in aug_args['video'].items():
            vid_trasnform.append(getattr(A, key.split('_')[0])(**val))
        return A.Compose(img_transform), A.ComposeV(vid_trasnform)

    def gen_samples_gopro(self, root_dir, video_list, suffix, num_gts, num_fut,
                          num_past):
        samples = []
        video_dirs = video_list if isinstance(video_list, list) else [
            video_list]

        inp_fmt = '{:06d}.' + suffix
        gt_fmt = '{:06d}.' + suffix

        print(f"[DEBUG] gen_samples_gopro called with root_dir={root_dir}")
        print(f"[DEBUG] video_dirs={video_dirs}")

        for vid_dir in video_dirs:
            inp_dir_path = join(root_dir, vid_dir, 'blur')
            gt_dir_path = join(root_dir, vid_dir, 'sharp')

            print(
                f"[DEBUG] Checking {vid_dir}: inp={inp_dir_path}, exists={os.path.exists(inp_dir_path)}")

            if not os.path.exists(inp_dir_path) or not os.path.exists(
                    gt_dir_path):
                print(f"[DEBUG] SKIP - directories don't exist")
                continue

            try:
                blur_imgs = sorted([item for item in os.listdir(inp_dir_path) if
                                    item.endswith(suffix)])
                sharp_imgs = sorted([item for item in os.listdir(gt_dir_path) if
                                     item.endswith(suffix)])

                print(
                    f"[DEBUG] Found {len(blur_imgs)} blur, {len(sharp_imgs)} sharp")

                blur_indices = sorted(
                    [int(img.split('.')[0]) for img in blur_imgs])
                sharp_indices = sorted(
                    [int(img.split('.')[0]) for img in sharp_imgs])
            except Exception as e:
                print(f"[DEBUG] EXCEPTION: {e}")
                continue

            valid_indices = sorted(set(blur_indices) & set(sharp_indices))
            print(f"[DEBUG] Valid indices: {len(valid_indices)}")

            # if len(valid_indices) < 3:
            #     print(f"[DEBUG] SKIP - not enough frames")
            #     continue

            count = 0
            for i in range(1, len(valid_indices) - 1):
                frame_num = valid_indices[i]

                sample = {
                    'inp': [],
                    'gt': [],
                    'inp_anno': [],
                    'trend': [],
                    'flow': [],
                    'video': vid_dir
                }

                inp_path = join(inp_dir_path, inp_fmt.format(frame_num))
                gt_path = join(gt_dir_path, gt_fmt.format(frame_num))

                if exists(inp_path) and exists(gt_path):
                    sample['inp'].append(inp_path)
                    sample['gt'] += [gt_path] * num_gts
                    samples.append(sample)
                    count += 1

            print(f"[DEBUG] Added {count} samples from {vid_dir}")

        print(f"[DEBUG] TOTAL samples: {len(samples)}")
        return samples

    def check_samples(self, miss_file_list_path=None):
        miss_videos = set()
        for idx, sample in enumerate(self.samples):
            for img_path in sample['inp']:
                if not exists(img_path):
                    miss_videos.add(sample['video'])
                    print('missing file: ', img_path)
            for img_path in sample['gt']:
                if not exists(img_path):
                    miss_videos.add(sample['video'])
                    print('missing file: ', img_path)
            if self.use_trend:
                for img_path in sample['trend']:
                    if not exists(img_path):
                        miss_videos.add(sample['video'])
                        print('missing file: ', img_path)
            if self.use_flow:
                for img_path in sample['flow']:
                    if not exists(img_path):
                        miss_videos.add(sample['video'])
                        print('missing file: ', img_path)
            if idx % 1000 == 0:
                print('{} samples have been checked'.format(idx + 1))
        if miss_file_list_path is not None:
            miss_videos = list(miss_videos)
            with open(miss_file_list_path, 'w') as f:
                yaml.dump(miss_videos, f)

    def get_sample_idx(self, video_dir, idx):
        for i, sample in enumerate(self.samples):
            if join(video_dir, 'blur', '{:08d}.png'.format(idx)) in sample['inp'][len(sample['inp']) // 2]:
                return i
        return FileNotFoundError

    def load_sample(self, sample):
        """
        Load images (RGB) and resize to fit in GPU memory
        """
        tensor = {}
        target_size = (256, 256)

        try:
            # load and resize input images
            tensor['inp'] = []
            for img_path in sample['inp']:
                img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
                tensor['inp'].append(img)

            h, w, _ = tensor['inp'][0].shape

            # skip inp_bbox since GoPro doesn't have annotations
            tensor['inp_bbox'] = []

            # load and resize gt images
            tensor['gt'] = []
            for img_path in sample['gt']:
                img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
                tensor['gt'].append(img)

            if self.use_trend:
                tensor['trend'] = []
                for trend_path in sample['trend']:
                    trend = np.load(trend_path)
                    trend = cv2.resize(trend, target_size, interpolation=cv2.INTER_NEAREST)
                    tensor['trend'].append(trend)

            if self.use_flow:
                tensor['flow'] = []
                for flow_path in sample['flow']:
                    flow = np.load(flow_path)
                    flow_x = flow[:, :, 0::2]
                    flow_y = flow[:, :, 1::2]
                    flow_x = np.mean(flow_x, axis=-1, keepdims=True)
                    flow_y = np.mean(flow_y, axis=-1, keepdims=True)
                    flow = np.concatenate([flow_x, flow_y], axis=-1)
                    flow = cv2.resize(flow, target_size, interpolation=cv2.INTER_AREA)
                    tensor['flow'].append(flow)

            tensor['video'] = sample['video']
        except:
            return None
        return tensor

    def replay_image_aug(self, tensor, transform):
        """
        Replay the same augmentation to each images in the tensor
        """
        # img, bbox = tensor['inp'][-1], tensor['inp_bbox'][-1]

        # img = tensor['inp'][-1]
        # bbox = None  # GoPro has no bboxes, just set to None
        #
        # # out = transform(image=img, bbox=bbox)
        # out = img
        # tensor['inp'][-1], replay_args = out['image'], out['replay_args']
        # for i, img in enumerate(tensor['inp'][:-1]):
        #     tensor['inp'][i] = transform(image=img, bbox=bbox, replay_args=replay_args)['image']
        # for i, img in enumerate(tensor['gt']):
        #     tensor['gt'][i] = transform(image=img, bbox=bbox, replay_args=replay_args)['image']
        # if self.use_trend:
        #     for i, trend in enumerate(tensor['trend']):
        #         tensor['trend'][i] = transform(image=trend, bbox=bbox, replay_args=replay_args, trend=True)['image']
        # if self.use_flow:
        #     for i, flow in enumerate(tensor['flow']):
        #         tensor['flow'][i] = transform(image=flow, bbox=bbox, replay_args=replay_args, flow=True)['image']
        return tensor

    def replay_video_aug(self, tensor, transform):
        """
        Replay the same augmentation to videos
        """
        images = tensor['inp']
        out = transform(images=images)
        tensor['inp'], replay_args = out['images'], out['replay_args']
        tensor['gt'] = transform(images=tensor['gt'], replay_args=replay_args)['images']
        if self.use_trend:
            tensor['trend'] = transform(images=tensor['trend'], replay_args=replay_args, trend=True)['images']
        if self.use_flow:
            tensor['flow'] = transform(images=tensor['flow'], replay_args=replay_args, flow=True)['images']
        return tensor

    def add_noise(self, x, sigma=0.1):
        return x + torch.randn_like(x) * sigma

    def __getitem__(self, idx):
        """
        Return a tensor dict with an idx:
        tensor['inp'] has shape of (N, C, H, W)
        tensor['gt'] has shape of (num_gts * N, C, H, W)
        tensor['trend'] has shape of (N, 2, H, W)
        tensor['video'] is the name of the video dir that the tensors belong to
        """
        sample = self.samples[idx]
        tensor = self.load_sample(sample)
        while tensor is None:
            idx = idx + 1 if idx < len(self.samples) - 1 else 0
            sample = self.samples[idx]
            tensor = self.load_sample(sample)
        tensor = self.replay_image_aug(tensor, self.img_transform)
        tensor = self.replay_video_aug(tensor, self.vid_transform)
        if self.noisy:
            # print("noisy? ", self.noisy)
            # print("Adding noise with ", self.sigma)
            inp = [torch.from_numpy(img).float() for img in tensor['inp']]
            tensor['inp'] = [self.add_noise(img, self.sigma) for img in inp]
        tensor['inp'] = torch.from_numpy(np.stack(tensor['inp'], axis=0).transpose((0, 3, 1, 2))).float()
        tensor['gt'] = torch.from_numpy(np.stack(tensor['gt'], axis=0).transpose((0, 3, 1, 2))).float()
        if self.use_trend:
            tensor['trend'] = torch.from_numpy(np.stack(tensor['trend'], axis=0).transpose((0, 3, 1, 2))).float()
        if self.use_flow:
            tensor['flow'] = torch.from_numpy(np.stack(tensor['flow'], axis=0).transpose((0, 3, 1, 2))).float()

        return tensor

    def __len__(self):
        return len(self.samples)

    def visualize(self, idx, aug=False):
        """
        Visualize the images to check the correctness of implementation
        """
        sample = self.samples[idx]
        tensor = self.load_sample(sample)
        if aug:
            tensor = self.replay_image_aug(tensor, self.img_transform)
            tensor = self.replay_video_aug(tensor, self.vid_transform)
        imgs = []
        num_gts = len(tensor['gt']) // len(tensor['inp'])
        for i, inp_img in enumerate(tensor['inp']):
            if self.use_trend:
                # a row of images, including a input blurry image, trend guidance and corresponding gt sharp image sequence
                imgs.append(
                    np.concatenate([inp_img,
                                    trend_plus_vis(tensor['trend'][i]),
                                    *tensor['gt'][i * num_gts:(i + 1) * num_gts]], axis=1)
                )
            elif self.use_flow:
                # a row of images, including a input blurry image, optical flow and corresponding gt sharp image sequence
                imgs.append(
                    np.concatenate([inp_img,
                                    flow_to_image(tensor['flow'][i]),
                                    *tensor['gt'][i * num_gts:(i + 1) * num_gts]], axis=1)
                )
            else:
                # a row of images, including a input blurry image, and corresponding gt sharp image sequence
                imgs.append(
                    np.concatenate([inp_img,
                                    *tensor['gt'][i * num_gts:(i + 1) * num_gts]], axis=1)
                )
        imgs = np.concatenate(imgs, axis=0).astype(np.uint8)
        plt.figure(figsize=(8 * (imgs.shape[1] // imgs.shape[0]), 8))
        plt.axis('off')
        plt.title('sample from {}'.format(sample['video']))
        plt.imshow(imgs)
        plt.show()


class GenBlur(Dataset):
    """
    Dataset class of videos, each video must be represented as a dir
    Structure under each video dir:
        blur: {:08d}.png (start from 00000000.png)
        sharp: {:08d}_{:03d}.png (start from 00000000_000.png)
        trend+: {:08d}_trend.npy (start from 00000000_trend.npy)
    """

    def __init__(self, set_type, root_dir, suffix, num_gts, num_fut, num_past, video_list, aug_args, use_trend=False,
                 temporal_step=1, trend_type='trend+', check_trend=False, use_flow=False, flow_type='flow'):
        self.use_trend = use_trend
        self.trend_type = trend_type
        self.check_trend = check_trend
        self.use_flow = use_flow
        self.flow_type = flow_type
        self.img_transform, self.vid_transform = self.gen_transform(aug_args[set_type])
        assert isinstance(root_dir, list) and isinstance(video_list, dict)
        self.samples = []
        for sub_dir in root_dir:
            sub_list = video_list[basename(sub_dir)]
            self.samples += self.gen_samples(sub_dir, sub_list[set_type], suffix, num_gts, num_fut, num_past)
        self.samples = self.samples[::temporal_step]

    def gen_transform(self, aug_args):
        """
        Generate replay transform based on augmentation arguments in the config file
        """
        img_transform, vid_trasnform = [], []
        for key, val in aug_args['image'].items():
            img_transform.append(getattr(A, key.split('_')[0])(**val))
        for key, val in aug_args['video'].items():
            vid_trasnform.append(getattr(A, key.split('_')[0])(**val))
        return A.Compose(img_transform), A.ComposeV(vid_trasnform)

    def gen_samples(self, root_dir, video_list, suffix, num_gts, num_fut, num_past):
        """
        Returned samples is a list of dicts, a sample is represented as a dict
        sample['inp'] is a list of the file paths of temporally continuous input images
        sample['gt'] is a list of the file paths of gt images for the corresponding blurry input image
        sample['trend'] is a list of npy file paths, the 1/2X trend guidance
        sample['video'] is the name of the video dir that the sample belongs to
        """
        samples = []
        if isinstance(video_list, list):
            video_dirs = video_list
        else:
            with open(video_list) as f:
                video_dirs = yaml.full_load(f)
        inp_fmt = '{:08d}.' + suffix
        gt_fmt = '{:08d}_{:03d}.' + suffix
        trend_fmt = '{:08d}_trend.npy'
        flow_fmt = '{:08d}_flow.npy'
        for vid_dir in video_dirs:
            inp_dir_path = join(root_dir, vid_dir, 'blur')
            gt_dir_path = join(root_dir, vid_dir, 'sharp')
            trend_dir_path = join(root_dir, vid_dir, self.trend_type)
            flow_dir_path = join(root_dir, vid_dir, self.flow_type)
            try:
                imgs = [item for item in os.listdir(inp_dir_path) if item.endswith(suffix)]
            except:
                print("missing videos:", inp_dir_path)
                continue
            num_imgs = len(imgs)

            range_start = num_past
            range_stop = num_imgs - num_fut
            if range_start >= range_stop:
                continue

            for frame in range(range_start, range_stop):
                sample = {}
                sample['inp'], sample['gt'], sample['trend'], sample['flow'] = [], [], [], []

                for i in range(frame - num_past, frame + num_fut + 1):
                    # collect path of inputdataset.py
                    inp_path = join(inp_dir_path, inp_fmt.format(i))
                    sample['inp'].append(inp_path)
                    # collect path of gts
                    gt_paths = [join(gt_dir_path, gt_fmt.format(i, j)) for j in range(num_gts)]
                    sample['gt'] += gt_paths

                    if self.use_trend:
                        # collect trend guidance
                        trend_path = join(trend_dir_path, trend_fmt.format(i))
                        sample['trend'].append(trend_path)

                    if self.use_flow:
                        # collect optical flow
                        flow_path = join(flow_dir_path, flow_fmt.format(i))
                        sample['flow'].append(flow_path)

                sample['video'] = vid_dir
                samples.append(sample)

        return samples

    def check_samples(self, miss_file_list_path=None):
        miss_videos = set()
        for idx, sample in enumerate(self.samples):
            for img_path in sample['inp']:
                if not exists(img_path):
                    miss_videos.add(sample['video'])
                    print('missing file: ', img_path)
            for img_path in sample['gt']:
                if not exists(img_path):
                    miss_videos.add(sample['video'])
                    print('missing file: ', img_path)
            if self.use_trend:
                for img_path in sample['trend']:
                    if not exists(img_path):
                        miss_videos.add(sample['video'])
                        print('missing file: ', img_path)
            if self.use_flow:
                for img_path in sample['flow']:
                    if not exists(img_path):
                        miss_videos.add(sample['video'])
                        print('missing file: ', img_path)
            if idx % 1000 == 0:
                print('{} samples have been checked'.format(idx + 1))
        if miss_file_list_path is not None:
            miss_videos = list(miss_videos)
            with open(miss_file_list_path, 'w') as f:
                yaml.dump(miss_videos, f)

    def get_sample_idx(self, video_dir, idx):
        for i, sample in enumerate(self.samples):
            if join(video_dir, 'blur', '{:08d}.png'.format(idx)) in sample['inp'][len(sample['inp']) // 2]:
                return i
        return FileNotFoundError

    def load_sample(self, sample):
        """
        Load images (RGB) and trend guidance from the paths in the sample dict to store as tensor dict
        """
        tensor = {}
        try:
            # load data for input
            tensor['inp'] = [cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) for img_path in sample['inp']]
            h, w, _ = tensor['inp'][0].shape
            # load data for gt
            tensor['gt'] = [cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB) for img_path in sample['gt']]
            if self.use_trend:
                # load trend guidance
                tensor['trend'] = []
                for trend_path in sample['trend']:
                    trend = np.load(trend_path)
                    # recover the size of the trend as input image
                    trend = cv2.resize(trend, (w, h), interpolation=cv2.INTER_NEAREST)
                    tensor['trend'].append(trend)
            if self.use_flow:
                # load optical flow
                tensor['flow'] = []
                for flow_path in sample['flow']:
                    flow = np.load(flow_path)
                    flow_x = flow[:, :, 0::2]
                    flow_y = flow[:, :, 1::2]
                    flow_x = np.mean(flow_x, axis=-1, keepdims=True)
                    flow_y = np.mean(flow_y, axis=-1, keepdims=True)
                    flow = np.concatenate([flow_x, flow_y], axis=-1)
                    flow = cv2.resize(flow, (w, h), interpolation=cv2.INTER_AREA)  # (h, w, 2)
                    tensor['flow'].append(flow)
            # load video name
            tensor['video'] = sample['video']
        except:
            print(sample['inp'])
            return None
        return tensor

    def replay_image_aug(self, tensor, transform):
        """
        Replay the same augmentation to each images in the tensor
        """
        img, bbox = tensor['inp'][-1], None
        out = transform(image=img, bbox=bbox)
        tensor['inp'][-1], replay_args = out['image'], out['replay_args']
        for i, img in enumerate(tensor['inp'][:-1]):
            tensor['inp'][i] = transform(image=img, bbox=bbox, replay_args=replay_args)['image']
        for i, img in enumerate(tensor['gt']):
            tensor['gt'][i] = transform(image=img, bbox=bbox, replay_args=replay_args)['image']
        if self.use_trend:
            for i, trend in enumerate(tensor['trend']):
                tensor['trend'][i] = transform(image=trend, bbox=bbox, replay_args=replay_args, trend=True)['image']
        if self.use_flow:
            for i, flow in enumerate(tensor['flow']):
                tensor['flow'][i] = transform(image=flow, bbox=bbox, replay_args=replay_args, flow=True)['image']
        return tensor

    def replay_video_aug(self, tensor, transform):
        """
        Replay the same augmentation to videos
        """
        images = tensor['inp']
        out = transform(images=images)
        tensor['inp'], replay_args = out['images'], out['replay_args']
        tensor['gt'] = transform(images=tensor['gt'], replay_args=replay_args)['images']
        if self.use_trend:
            tensor['trend'] = transform(images=tensor['trend'], replay_args=replay_args, trend=True)['images']
        if self.use_flow:
            tensor['flow'] = transform(images=tensor['flow'], replay_args=replay_args, flow=True)['images']
        return tensor

    def check_trend_ratio(self, tensor, threshold=0.01, verbose=False):
        if tensor is None:
            return None
        trends = tensor['trend']
        trend = trends[len(trends) // 2]
        trend = np.sum(np.abs(trend), axis=-1)
        trend[trend > 0] = 1
        ratio = np.mean(trend)
        if ratio < threshold:
            if verbose:
                print('failed ratio {}'.format(ratio))
            return None
        else:
            return tensor

    def __getitem__(self, idx):
        """
        Return a tensor dict with an idx:
        tensor['inp'] has shape of (N, C, H, W)
        tensor['gt'] has shape of (num_gts * N, C, H, W)
        tensor['trend'] has shape of (N, 2, H, W)
        tensor['video'] is the name of the video dir that the tensors belong to
        """
        sample = self.samples[idx]
        tensor = self.load_sample(sample)  # deal with None tensor
        if self.check_trend:
            tensor = self.check_trend_ratio(tensor)
        while tensor is None:
            idx = idx + 1 if idx < len(self.samples) - 1 else 0
            sample = self.samples[idx]
            tensor = self.load_sample(sample)
            if self.check_trend:
                tensor = self.check_trend_ratio(tensor)
        # tensor = self.check_trend_ratio(tensor, verbose=True)
        tensor = self.replay_image_aug(tensor, self.img_transform)
        tensor = self.replay_video_aug(tensor, self.vid_transform)
        tensor['inp'] = torch.from_numpy(np.stack(tensor['inp'], axis=0).transpose((0, 3, 1, 2))).float()
        tensor['gt'] = torch.from_numpy(np.stack(tensor['gt'], axis=0).transpose((0, 3, 1, 2))).float()
        if self.use_trend:
            tensor['trend'] = torch.from_numpy(np.stack(tensor['trend'], axis=0).transpose((0, 3, 1, 2))).float()
        if self.use_flow:
            tensor['flow'] = torch.from_numpy(np.stack(tensor['flow'], axis=0).transpose((0, 3, 1, 2))).float()

        return tensor

    def __len__(self):
        return len(self.samples)

    def visualize(self, idx, aug=False):
        """
        Visualize the images to check the correctness of implementation
        """
        sample = self.samples[idx]
        tensor = self.load_sample(sample)
        if aug:
            tensor = self.replay_image_aug(tensor, self.img_transform)
            tensor = self.replay_video_aug(tensor, self.vid_transform)
        imgs = []
        num_gts = len(tensor['gt']) // len(tensor['inp'])
        for i, inp_img in enumerate(tensor['inp']):
            if self.use_trend:
                # a row of images, including a input blurry image, trend guidance and corresponding gt sharp image sequence
                imgs.append(
                    np.concatenate([inp_img,
                                    trend_plus_vis(tensor['trend'][i]),
                                    *tensor['gt'][i * num_gts:(i + 1) * num_gts]], axis=1)
                )
            elif self.use_flow:
                # a row of images, including a input blurry image, optical flow and corresponding gt sharp image sequence
                imgs.append(
                    np.concatenate([inp_img,
                                    flow_to_image(tensor['flow'][i]),
                                    *tensor['gt'][i * num_gts:(i + 1) * num_gts]], axis=1)
                )
            else:
                # a row of images, including a input blurry image, and corresponding gt sharp image sequence
                imgs.append(
                    np.concatenate([inp_img,
                                    *tensor['gt'][i * num_gts:(i + 1) * num_gts]], axis=1)
                )
        imgs = np.concatenate(imgs, axis=0).astype(np.uint8)
        plt.figure(figsize=(8 * (imgs.shape[1] // imgs.shape[0]), 8))
        plt.axis('off')
        plt.title('sample from {}'.format(sample['video']))
        plt.imshow(imgs)
        plt.show()


if __name__ == '__main__':
    # test code for BAistPP
    os.chdir('..')
    use_trend = False
    use_flow = True
    with open('./configs/base.yaml') as f:
        config = yaml.full_load(f)
    dataset_args = config['dataset_args']
    dataset_args['num_fut'] = 0
    dataset_args['num_past'] = 0
    # dataset_args['root_dir'] = [
    #     '../netdata/videos/gopro',
    #     '../netdata/videos/mygopro',
    #     '../netdata/videos/adobe'
    # ]
    dataset_args['root_dir'] = [
        '../netdata/videos_demo/b-aist++',
    ]
    dataset_args['video_list']['b-aist++']['train'] = ['gHO_sBM_c01_d20_mHO0_ch03_cropped_32X', ]
    dataset = BAistPP('train', **dataset_args, use_trend=use_trend, use_flow=use_flow)
    print('number of samples for one epoch:', len(dataset))
    idx = 123
    idx = idx % len(dataset)
    tensor = dataset[idx]
    print('shape of input image:', tuple(tensor['inp'].shape))
    print('shape of gt tensor:', tuple(tensor['gt'].shape))
    if use_trend:
        print('shape of trend tensor:', tuple(tensor['trend'].shape))
    if use_flow:
        print('shape of flow tensor:', tuple(tensor['flow'].shape))
    print('name of the video dir that the sample belongs to:', tensor['video'])
    dataset.visualize(idx, aug=True)