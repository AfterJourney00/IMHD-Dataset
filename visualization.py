import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt

from pytorch3d.transforms import axis_angle_to_matrix
from pytorch3d.structures import Meshes, join_meshes_as_scene
from pytorch3d.renderer import TexturesVertex
from pytorch3d.io import load_obj
from pytorch3d.utils import cameras_from_opencv_projection
from pytorch3d.renderer import (
    RasterizationSettings, MeshRenderer, MeshRasterizer, SoftPhongShader
)
 
import pickle as pkl
import os, json, argparse

from body_model.body_model import BodyModel

# Set the cuda device
if torch.cuda.is_available():
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
else:
    device = torch.device("cpu")

# constants
data_root = r'data/'
SMPL_MODEL_DIR = r'./body_model/smplh/neutral/model.npz'
smplh = BodyModel(SMPL_MODEL_DIR, num_betas=16).to(device)

def call_args():
    parser = argparse.ArgumentParser(description="Visualize Ground Truth.")
    parser.add_argument('--motion', type=str, default=r'20230910/20230910_af_skateboard/freestyle1/gt_0_338_2309.pkl')
    parser.add_argument('--object', type=str, default='skateboard')
    parser.add_argument('--save_dir', default=r'visualizations')
    parser.add_argument('--render_width', type=int, default=1920)
    parser.add_argument('--render_height', type=int, default=1080)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=-1)
    parser.add_argument('--no_motion', action='store_true')
    parser.add_argument('--no_imu', action='store_true')
    args = parser.parse_args()

    date = args.motion.split('/')[0]
    args.save_dir = os.path.join(args.save_dir, '/'.join(args.motion.split('/')[:-1]))
    args.save_motion_dir = os.path.join(args.save_dir, 'motion')
    args.save_imu_dir = os.path.join(args.save_dir, 'imu')
    args.motion = os.path.join(data_root, 'ground_truth', args.motion)
    args.imu = args.motion.replace('ground_truth', 'imu_preprocessed').replace('/gt_', '/imu_')
    args.object = os.path.join(data_root, 'object_templates', args.object, args.object+'_simplified_transformed.obj')
    args.extrin = os.path.join(data_root, 'calibrations', date, 'extrin.json')
    args.intrin = os.path.join(data_root, 'calibrations', date, 'intrin.json')

    assert args.render_width / args.render_height == 16 / 9, "Please specify the render resolution complying with 16:9."
    os.makedirs(args.save_motion_dir, exist_ok=True)
    os.makedirs(args.save_imu_dir, exist_ok=True)

    return args

def load_cameras(args):
    with open(args.extrin, 'r') as f:
        extrin_data = json.load(f)
    with open(args.intrin, 'r') as f:
        intrin_data = json.load(f)["color"]

    R = np.array(extrin_data['rotation']).reshape((1, 3, 3))
    T = np.array(extrin_data['translation']).reshape((1, 3))
    K = [intrin_data['fx'], 0, intrin_data['cx'], 0, intrin_data['fy'], intrin_data['cy'], 0, 0, 1]
    K = np.array(K).reshape((3, 3))

    resolution_ratio = 3840 / args.render_width # raw images are 4K
    K = K / resolution_ratio
    K[2, 2] = 1
    
    return R, T, K

def make_renderer(R, T, K, args):
    # define rasterization setting
    cameras = cameras_from_opencv_projection(
        R,
        T,
        K,
        torch.tensor([args.render_height, args.render_width]).unsqueeze(0)
    )

    # define rasterization setting
    raster_settings = RasterizationSettings(
        image_size=[args.render_height, args.render_width],
        blur_radius=0.0,
        faces_per_pixel=10,
        max_faces_per_bin=100000
    )

    # define shader
    bp = None
    shader = SoftPhongShader(
        device=device,
        cameras=cameras,
        blend_params=bp
    )

    # create renderer
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=shader
    )

    return renderer

def transform_template(objVerts, rot, trans):
    trans = trans.to(device)
    rot = rot.to(device)
    rot = axis_angle_to_matrix(rot).view(3,3)

    return torch.mm(objVerts.to(device), rot.T) + trans

def render(args):
    # load motion and object
    with open(args.motion, 'rb') as f:
        gt_data = pkl.load(f)
    with open(args.object, 'r') as f:
        templateVerts, objFaces, _ = load_obj(f)
    templateVerts -= torch.mean(templateVerts, dim=0)
    objFaces = objFaces.verts_idx

    # specify vertex color
    obj_rgb = torch.ones_like(templateVerts)
    smplh_rgb = torch.ones(6890, 3)
    verts_texture = TexturesVertex(verts_features=[obj_rgb.to(device), smplh_rgb.to(device)])


    # load camera parameters
    R, T, K = load_cameras(args)

    # create renderer
    renderer = make_renderer(
        torch.from_numpy(R).float().to(device),
        torch.from_numpy(T).float().to(device),
        torch.from_numpy(K).unsqueeze(0).float().to(device),
        args
    )

    # render
    end = gt_data['objectRot'].shape[0] if args.end == -1 else args.end + 1
    for frame in range(args.start, end):
        objectRot = torch.from_numpy(gt_data['objectRot'][frame]).unsqueeze(0).float()
        objectTrans = torch.from_numpy(gt_data['objectTrans'][frame]).unsqueeze(0).float()
        objVerts = transform_template(templateVerts, objectRot, objectTrans)

        smplPose = torch.from_numpy(gt_data['smplPose'][frame]).unsqueeze(0).float()
        smplHandPose = torch.from_numpy(gt_data['smplHandPose'][frame]).unsqueeze(0).float()
        smplShape = torch.from_numpy(gt_data['smplShape'][frame]).unsqueeze(0).float()
        smplTrans = torch.from_numpy(gt_data['smplTrans'][frame]).unsqueeze(0).float()
        smplGt = smplh(
            betas=smplShape.to(device),
            root_orient=smplPose[:, :3].to(device),
            pose_body=smplPose[:, 3:66].to(device),
            pose_hand=smplHandPose.to(device),
            trans=smplTrans.to(device)
        )
        humanVerts = smplGt.v.squeeze(0)
        humanFaces = smplGt.f
        

        ho_mesh = Meshes(
            verts=[objVerts.to(device), humanVerts.to(device)],
            faces=[objFaces.to(device), humanFaces.to(device)],
            textures=verts_texture
        )

        rendered_img = renderer(join_meshes_as_scene(ho_mesh))
        cv2.imwrite(
            os.path.join(args.save_motion_dir, f'{frame}.png'),
            255 * rendered_img[0, :, :, :-1].detach().cpu().numpy()
        )

def plot(id, x, y, args, **kwargs):
    plt.subplot(1, 2, id)
    plt.plot(x, y[0], color='red', label='x', marker='o', mfc='w', ms=2.5, markevery=kwargs["mark"])
    plt.plot(x, y[1], color='green', label='y', marker='o', mfc='w', ms=2.5, markevery=kwargs["mark"])
    plt.plot(x, y[2], color='blue', label='z', marker='o', mfc='w', ms=2.5, markevery=kwargs["mark"])
    plt.xlim((args.start, kwargs["end"]))
    plt.ylim((kwargs["lb"] - 0.05, kwargs["ub"] + 0.05))
    plt.grid(True, linestyle='--')
    plt.legend(fancybox=True, framealpha=0.5, loc='upper right')

def plot_imu(args):
    with open(args.imu, 'rb') as f:
        imu_data = pkl.load(f)
    rot_data = imu_data['objectImuOri']; acc_data = imu_data['objectImuAcc']
    rot_ub = np.max(rot_data); rot_lb = np.min(rot_data)
    acc_ub = np.max(acc_data); acc_lb = np.min(acc_data)

    end = rot_data.shape[0] if args.end == -1 else args.end + 1
    duration = end - args.start + 1
    gap = duration // 64
    mark_per = gap if gap > 0 else None
    
    x = np.arange(start=args.start, stop=end, step=1)
    plot_rot = rot_data[args.start:end].T; plot_acc = acc_data[args.start:end].T

    plt.figure(figsize=(16, 4))
    plt.rc('font', family='serif')
    plot(1, x, plot_rot, args, end=end, lb=rot_lb, ub=rot_ub, mark=mark_per); plt.title("rot")
    plot(2, x, plot_acc, args, end=end, lb=acc_lb, ub=acc_ub, mark=mark_per); plt.title("acc")
    plt.suptitle("Imu Signal Visualization")
    plt.savefig(
        os.path.join(args.save_imu_dir, f'imu.png'),
        bbox_inches='tight',
        dpi=600
    )
    plt.close()

def main():
    args = call_args()

    if not args.no_imu:
        plot_imu(args)
    if not args.no_motion:
        render(args)

if __name__ == "__main__":
    main()