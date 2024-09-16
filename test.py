import torch
with open("D:\Workspace\Project_final_2/fed_global/fashion-mnist\log_main_iid\save_eval\evaluation_file.pkl", "rb") as f:
    data_loaded = torch.load(f)
print(data_loaded)

# with open("D:\Workspace\Project_final_2/fed_client\mnist\log_main_iid\save_eval\evaluation_file.pkl",
#           "rb") as f:
#     ssim_dict, mse_dict, pixel_dict, avg_ssim_t, avg_mse_t, avg_pixel_t = torch.load(f)
# print(f"ssim_dict: {ssim_dict}\nmse_dict: {mse_dict}\npixel_dict: {pixel_dict}\n"
#       f"avg_ssim_t: {avg_ssim_t}\navg_mse_t: {avg_mse_t}\navg_pixel_t: {avg_pixel_t}")