from utils.radar import *
from utils.lidar import *
from utils.utils import *
from utils.config import ox_dataroot


if __name__ == '__main__':
    # Check the config.py file in the utils folder! Set the oxford_dataroot in the file
    dataroot = ox_dataroot

    # Available sequences in the dataroot
    sequences = sorted([os.path.join(dataroot, f) for f in os.listdir(dataroot) if os.path.isdir(os.path.join(dataroot, f))])

    # Required BEV image size and resolution
    bev_img_size = 256
    bev_img_res = 0.5

    for seq_dir in sequences:
        print(f"Processing sequence: {seq_dir}")

        # Location where radar polar images are stored
        polar_radar_path = f"{seq_dir}/radar"

        # Directory to store the BEV images
        radar_results = f"{seq_dir}/radar_bev"
        os.makedirs(radar_results, exist_ok=True)

        for file in tqdm(os.listdir(polar_radar_path)):
            if not file.endswith('.png'):
                continue

            # Polar radar image path            
            radar_im_path = os.path.join(polar_radar_path, file)

            # Process radar data using library functions and get bev image
            timestamps, azimuths, valid, fft_data, radar_resolution = load_radar(radar_im_path)
            radar_img = radar_polar_to_cartesian(azimuths, fft_data, radar_resolution, bev_img_res, bev_img_size)

            radar_img = radar_img[:, :, 0]

            # Scale pixel information to [0, 255]
            radar_img = ((radar_img - np.min(radar_img)) * (1/(np.max(radar_img) -
                                                            np.min(radar_img))) * 255).astype('uint8')

            # Store BEV image in the save location
            cv2.imwrite(os.path.join(radar_results, file), radar_img.astype(np.uint8))





