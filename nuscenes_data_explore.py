from nuscenes.nuscenes import NuScenes
import os
import cv2
from pprint import pprint
from nuscenes_fusion import project_cam_coords_to_pixel, compute_3d_box_cam_coords_nuscenes
from alfred.fusion.common import draw_3d_box
from alfred.vis.image.common import get_unique_color_by_id
from nuscenes.utils.data_classes import Box


os.makedirs('results', exist_ok=True)

NameMapping = {
        'movable_object.barrier': 'barrier',
        'vehicle.bicycle': 'bicycle',
        'vehicle.bus.bendy': 'bus',
        'vehicle.bus.rigid': 'bus',
        'vehicle.car': 'car',
        'vehicle.construction': 'construction_vehicle',
        'vehicle.motorcycle': 'motorcycle',
        'human.pedestrian.adult': 'pedestrian',
        'human.pedestrian.child': 'pedestrian',
        'human.pedestrian.construction_worker': 'pedestrian',
        'human.pedestrian.police_officer': 'pedestrian',
        'human.pedestrian.wheelchair': 'pedestrian',
        'human.pedestrian.stroller': 'pedestrian',
        'human.pedestrian.personal_mobility': 'pedestrian',
        'movable_object.trafficcone': 'traffic_cone',
        'movable_object.pushable_pullable': 'pushable_pullable',
        'movable_object.debris': 'debris',
        'vehicle.trailer': 'trailer',
        'vehicle.truck': 'truck',
        'vehicle.emergency.ambulance': 'ambulance',
        'vehicle.emergency.police': 'police',
        'animal': 'animal',
        'static_object.bicycle_rack': 'bicycle_rack',
    }

def get_image(sample, root_dir):
    f_p = nusc.get('sample_data', sample['data']['CAM_FRONT'])
    return cv2.imread(os.path.join(root_dir, f_p['filename']))

root_dir = '../v1.0-trainval01'
nusc = NuScenes(version='v1.0-trainval', dataroot=root_dir, verbose=False)
all_category_db = list(NameMapping.keys())
all_category = list(NameMapping.values())

my_scene = nusc.scene[15]
sample_token = my_scene['first_sample_token']
end_token = my_scene['last_sample_token']
my_sample = nusc.get('sample', sample_token)

i = 0
while sample_token != end_token:
    my_sample = nusc.get('sample', my_sample['next'])
 
    img = get_image(my_sample, root_dir)
    sample_data = nusc.get_sample_data(my_sample['data']['CAM_FRONT'])
    labels = sample_data[1]
    # this label is according to camera, what we have is lidar
    # converts top lidar 
    intrinsic = sample_data[2]
    for label in labels:                 
        c2d = project_cam_coords_to_pixel([label.center], intrinsic)[0]
        cv2.circle(img, (int(c2d[0]), int(c2d[1])), 3, (0, 255, 255), -1)
        # convert center and wlh to 3d box
        box = Box(center=label.center, size=label.wlh, orientation=label.orientation)
        corners3d = box.corners()
        corners3d_2d = project_cam_coords_to_pixel(corners3d, intrinsic)

        idx = all_category_db.index(label.name)
        c = get_unique_color_by_id(idx)
        draw_3d_box(corners3d_2d, img, c)
        if len(corners3d_2d) > 4:
            cv2.putText(img, '{0}'.format(all_category[idx]),
                        (int(corners3d_2d[1][0]), int(corners3d_2d[1][1])), cv2.FONT_HERSHEY_PLAIN, .9, (255, 255, 255))

    # show some image and lidar points
    cv2.imshow('aa', img)
    cv2.imwrite('results/{}.png'.format(i), img)
    i += 1
    cv2.waitKey(0)
