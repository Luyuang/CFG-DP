# In plot.py, modify the draw_predicted_result function
def draw_predicted_result(task_name, all_img, upBody_lowDim, eps_idx, slice_idx=None):
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    # Set Times New Roman font
    times_path = "/usr/share/fonts/truetype/msttcorefonts/Times_New_Roman.ttf"
    prop = font_manager.FontProperties(fname=times_path)

    img_strips = []
    show_img_num = 18
    for img_name, img_eps in all_img.items():
        img_strip = np.concatenate(np.array(img_eps[eps_idx])[::max(1, len(img_eps[eps_idx])//show_img_num)], axis=1)
        img_strips.append(img_strip)

    img_strip_combined = np.vstack(img_strips)
    
    JOINT_DIM_LABELS = ["zarm_l1_link", "zarm_l2_link", "zarm_l3_link", "zarm_l4_link", "zarm_l5_link", "zarm_l6_link", "zarm_l7_link", 
                        "dex_hand_l1_link", "dex_hand_l2_link", "dex_hand_l3_link", "dex_hand_l4_link", "dex_hand_l5_link", "dex_hand_l6_link", 
                        "zarm_r1_link", "zarm_r2_link", "zarm_r3_link", "zarm_r4_link", "zarm_r5_link", "zarm_r6_link", "zarm_r7_link", 
                        "dex_hand_r1_link", "dex_hand_r2_link", "dex_hand_r3_link", "dex_hand_r4_link", "dex_hand_r5_link", "dex_hand_r6_link"]
    
    img_rows = len(all_img)
    figure_layout = [
        JOINT_DIM_LABELS[0:4],
        JOINT_DIM_LABELS[4:8],
        JOINT_DIM_LABELS[8:12],
        JOINT_DIM_LABELS[12:16],
        JOINT_DIM_LABELS[16:20],
        JOINT_DIM_LABELS[20:24],
        JOINT_DIM_LABELS[24:] + ['extra1', 'extra2'],
    ]
    for i in range(img_rows):
        row_images = ['image'] * 4
        figure_layout.insert(i, row_images)

    plt.rcParams.update({'font.size': 9, 'font.family': 'Times New Roman'})
    fig, axs = plt.subplot_mosaic(figure_layout)
    fig.set_size_inches([20, 18])
    fig.suptitle(task_name, fontsize=15, fontproperties=prop)
    
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
        axs[action_label].set_xlabel(action_label, labelpad=10, fontproperties=prop)
        if action_dim == 0:
            axs[action_label].legend(loc='best', fontsize=8, prop=prop)

    axs['image'].imshow(img_strip_combined)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.subplots_adjust(hspace=0.7)
    plt.show()

    # Plot Lambda Curve
    fig_lamda, ax_lamda = plt.subplots(figsize=(8, 3))
    lamda_arr = np.array(upBody_lowDim["lamda"][eps_idx])
    ax_lamda.plot(lamda_arr, label="Lambda", color="orange")
    ax_lamda.set_title("Lambda Curve", fontproperties=prop)
    ax_lamda.set_xlabel("Step", fontproperties=prop)
    ax_lamda.set_ylabel("Lambda", fontproperties=prop)
    ax_lamda.legend(prop=prop)
    plt.show()

    # Plot Conditional Entropy Curve
    fig_entropy, ax_entropy = plt.subplots(figsize=(8, 3))
    entropy_arr = np.array(upBody_lowDim["entropy"][eps_idx])
    ax_entropy.plot(entropy_arr, label="Conditional Entropy", color="blue")
    ax_entropy.set_title("Conditional Entropy of Action Distribution", fontproperties=prop)
    ax_entropy.set_xlabel("Timestep", fontproperties=prop)
    ax_entropy.set_ylabel("Entropy (bits)", fontproperties=prop)
    ax_entropy.legend(prop=prop)
    plt.grid(True)
    plt.show()