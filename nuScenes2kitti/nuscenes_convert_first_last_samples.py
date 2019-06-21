# convertion from NuScenes dataset to KITTI format
# inspired by https://github.com/poodarchu/nuscenes_to_kitti_format
# converting only camera captions (JPG files)
# converting only the first and last samples in every sequence data

# regardless of attributes indexed 2(if blocked) in KITTI label
# however, object minimum visibility level is adjustable


start_index = 6000
data_root = '/home/charlie/Downloads/v1.0-mini/'
img_output_root = './output/image/'
label_output_root = './output/label/'

min_visibility_level = '2'
delete_dontcare_objects = True


from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import BoxVisibility, view_points
import numpy as np
import cv2
import os
import shutil


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
    sensor_list = ['CAM_FRONT', 'CAM_BACK', 'CAM_FRONT_LEFT', 'CAM_BACK_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK_RIGHT']

    frame_counter = start_index

    if os.path.isdir(img_output_root) == True:
        print('previous image output found. deleting...')
        shutil.rmtree(img_output_root)
    os.makedirs(img_output_root)
    if os.path.isdir(label_output_root) == True:
        print('previous label output found. deleting...')
        shutil.rmtree(label_output_root)
    os.makedirs(label_output_root)

    # 10 scenes in mini set; 850 scenes in the whole set
    for present_scene in nusc.scene:
        first_sample_token = present_scene['first_sample_token']
        last_sample_token = present_scene['last_sample_token']
        first_sample = nusc.get('sample', first_sample_token)
        last_sample = nusc.get('sample', last_sample_token)
        sample_list = [first_sample, last_sample]

        # converting image data from 6 cameras (in the sensor list)
        for present_sensor in sensor_list:
            for present_sample in sample_list:

                # each sensor_data corresponds to one specific image in the dataset
                sensor_data = nusc.get('sample_data', present_sample['data'][present_sensor])
                data_path, box_list, cam_intrinsic = nusc.get_sample_data(present_sample['data'][present_sensor], BoxVisibility.ALL)

                img_file = data_root + sensor_data['filename']
                seqname = str(frame_counter).zfill(6)
                output_label_file = label_output_root + seqname + '.txt'

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
                    cmd = 'cp ' + img_file + ' ' + img_output_root + seqname + '.jpg'
                    print('copying', sensor_data['filename'], 'to', seqname + '.jpg')
                    os.system(cmd)
                    frame_counter += 1
