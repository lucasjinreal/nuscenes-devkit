# convertion from NuScenes dataset to KITTI format
# inspired by https://github.com/poodarchu/nuscenes_to_kitti_format
# converting only camera captions (JPG files)
# converting all samples in every sequence data

# regardless of attributes indexed 2(if blocked) in KITTI label
# however, object minimum visibility level is adjustable
"""

As for every camera the intrinsic is different, so that we put CAM_FRONT, CAM_LEFT, CAM_RIGHT
to different 


"""

from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import BoxVisibility, view_points
import numpy as np
import cv2
import os
import shutil



start_index = 0
data_root = '../../v1.0mini'

img_i_output_root = os.path.join(data_root, 'kitti', 'object', 'image_{}')
label_i_output_root = os.path.join(data_root, 'kitti', 'object', 'label_{}')

min_visibility_level = '2'
delete_dontcare_objects = True

category_reflection = \
{
    'human.pedestrian.adult': 'Pedestrian',
    'human.pedestrian.child': 'Pedestrian',
    'human.pedestrian.wheelchair': 'DontCare',
    'human.pedestrian.stroller': 'DontCare',
    'human.pedestrian.personal_mobility': 'DontCare',
    'human.pedestrian.police_officer': 'Pedestrian',
    'human.pedestrian.construction_worker': 'Pedestrian',
    'animal': 'DontCare',
    'vehicle.car': 'Car',
    'vehicle.motorcycle': 'Cyclist',
    'vehicle.bicycle': 'Cyclist',
    'vehicle.bus.bendy': 'Tram',
    'vehicle.bus.rigid': 'Tram',
    'vehicle.truck': 'Truck',
    'vehicle.construction': 'DontCare',
    'vehicle.emergency.ambulance': 'DontCare',
    'vehicle.emergency.police': 'DontCare',
    'vehicle.trailer': 'Tram',
    'movable_object.barrier': 'DontCare',
    'movable_object.trafficcone': 'DontCare',
    'movable_object.pushable_pullable': 'DontCare',
    'movable_object.debris': 'DontCare',
    'static_object.bicycle_rack': 'DontCare', 
}


if __name__ == '__main__':
    nusc = NuScenes(version='v1.0-mini', dataroot=data_root, verbose=True)
    sensor_list = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']

    frame_counter = start_index

    
    for present_sample in nusc.sample:
        # converting image data from 6 cameras (in the sensor list)
        for idx, present_sensor in enumerate(sensor_list):
            img_output_root = img_i_output_root.format(idx)
            label_output_root = label_i_output_root.format(idx)
            print('SENSOR: {}, img out: {}, label out: {}, idx: {}'.format(present_sensor, 
            img_output_root, label_output_root, idx))

            os.makedirs(img_output_root, exist_ok=True)
            os.makedirs(label_output_root, exist_ok=True)

            # each sensor_data corresponds to one specific image in the dataset
            sensor_data = nusc.get('sample_data', present_sample['data'][present_sensor])
            data_path, box_list, cam_intrinsic = nusc.get_sample_data(present_sample['data'][present_sensor], BoxVisibility.ALL)

            img_file = os.path.join(data_root, sensor_data['filename']) 
            seqname = str(frame_counter).zfill(6)
            output_label_file = os.path.join(label_output_root, seqname + '.txt')

            with open(output_label_file, 'a') as output_f:
                for box in box_list:
                    # obtaining visibility level of each 3D box
                    present_visibility_token = nusc.get('sample_annotation', box.token)['visibility_token']
                    if present_visibility_token > min_visibility_level:
                        if not (category_reflection[box.name] == 'DontCare' and delete_dontcare_objects):
                            w, l, h = box.wlh
                            x, y, z = box.center
                            yaw, pitch, roll = box.orientation.yaw_pitch_roll; yaw = -yaw
                            alpha = yaw - np.arctan2(x, z)
                            box_name = category_reflection[box.name]
                            # projecting 3D points to image plane
                            points_2d = view_points(box.corners(), cam_intrinsic, normalize=True)
                            left_2d = int(min(points_2d[0]))
                            top_2d = int(min(points_2d[1]))
                            right_2d = int(max(points_2d[0]))
                            bottom_2d = int(max(points_2d[1]))

                            line = "{} {} {} {} {} {} {} {} {} {} {} {} {} {} {}\n".format(
                                box_name, 0, -1, alpha, left_2d, top_2d, right_2d, bottom_2d, h, w, l, x, y+h/2, z, yaw)
                            output_f.write(line)

            if not os.path.getsize(output_label_file):
                del_cmd = 'rm ' + output_label_file
                os.system(del_cmd)
            else:
                cmd = 'cp ' + img_file + ' ' + os.path.join(img_output_root, seqname + '.jpg')
                print('copying', sensor_data['filename'], 'to', os.path.join(img_output_root, seqname + '.jpg'))
                os.system(cmd)
                frame_counter += 1
