def draw_predicted_result(task_name, all_img, upBody_lowDim, eps_idx, slice_idx = None):
    import numpy as np
    import matplotlib.pyplot as plt
    img_strips = []
    show_img_num = 18
    for img_name, img_eps in all_img.items():
        img_strip = np.concatenate(np.array(img_eps[eps_idx])[::len(img_eps[eps_idx])//show_img_num], axis=1)
        img_strips.append(img_strip)

    img_strip_combined = np.vstack(img_strips)
    
    JOINT_DIM_LABELS = ["zarm_l1_link", "zarm_l2_link", "zarm_l3_link", "zarm_l4_link", "zarm_l5_link", "zarm_l6_link", "zarm_l7_link", 
                        "dex_hand_l1_link", "dex_hand_l2_link", "dex_hand_l3_link", "dex_hand_l4_link", "dex_hand_l5_link", "dex_hand_l6_link", 
                        "zarm_r1_link", "zarm_r2_link", "zarm_r3_link", "zarm_r4_link", "zarm_r5_link", "zarm_r6_link", "zarm_r7_link", 
                        "dex_hand_r1_link", "dex_hand_r2_link", "dex_hand_r3_link", "dex_hand_r4_link", "dex_hand_r5_link", "dex_hand_r6_link"]
    
    img_rows = len(all_img)
    # Modified to have 8 subplots per row
    figure_layout = [
        JOINT_DIM_LABELS[0:8],  # First 8 joints
        JOINT_DIM_LABELS[8:16],  # Next 8 joints
        JOINT_DIM_LABELS[16:24],  # Next 8 joints
        JOINT_DIM_LABELS[24:] + ['extra1', 'extra2', 'extra3', 'extra4', 'extra5', 'extra6'],  # Last 2 joints + 6 placeholders
    ]
    for i in range(img_rows):
        row_images = ['image'] * 8  # Changed to 8 'image' labels per row
        figure_layout.insert(i, row_images)

    plt.rcParams.update({'font.size': 7})
    fig, axs = plt.subplot_mosaic(figure_layout)
    fig.set_size_inches([32, 15])  # Slightly wider to accommodate 8 subplots
    fig.suptitle(task_name, fontsize=15)
    
    if slice_idx is not None:
        SELECTED_JOINT = [x for slc in slice_idx for x in JOINT_DIM_LABELS[slc[0]:slc[1]]]
    else:
        SELECTED_JOINT = JOINT_DIM_LABELS[0:8] + JOINT_DIM_LABELS[13:21] 
    
    for action_dim, action_label in enumerate(SELECTED_JOINT):
        for low_dim_name, low_dim_values in upBody_lowDim.items():
            if 'pred' in low_dim_name:  
                axs[action_label].plot(low_dim_values[eps_idx][:, 0, action_dim], label=low_dim_name, alpha=1, zorder=1)
            elif 'true_actions' in low_dim_name:
                axs[action_label].plot(low_dim_values[eps_idx][:, action_dim], label=low_dim_name, alpha=0.5, zorder=1)
            elif 'true_states' in low_dim_name:
                axs[action_label].plot(low_dim_values[eps_idx][:, action_dim], label=low_dim_name, alpha=0.2, zorder=1)
        axs[action_label].set_xlabel(action_label, labelpad=5)
        axs[action_label].legend()

    axs['image'].imshow(img_strip_combined)
    plt.legend()
    plt.show()