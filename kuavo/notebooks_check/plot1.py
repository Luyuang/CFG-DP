def draw_predicted_result(task_name, all_img, upBody_lowDim, eps_idx, slice_idx = None):
    import numpy as np
    import matplotlib.pyplot as plt
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
    # 每行4个子图
    figure_layout = [
        JOINT_DIM_LABELS[0:4],
        JOINT_DIM_LABELS[4:8],
        JOINT_DIM_LABELS[8:12],
        JOINT_DIM_LABELS[12:16],
        JOINT_DIM_LABELS[16:20],
        JOINT_DIM_LABELS[20:24],
        JOINT_DIM_LABELS[24:] + ['extra1', 'extra2'],  # 最后一行补齐到4个
    ]
    for i in range(img_rows):
        row_images = ['image'] * 4  # 每行4个image
        figure_layout.insert(i, row_images)

    plt.rcParams.update({'font.size': 9})
    fig, axs = plt.subplot_mosaic(figure_layout)
    fig.set_size_inches([20, 18])  # 调整宽高适配4列
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
        axs[action_label].set_xlabel(action_label, labelpad=10)
        # 只在第一个子图显示图例
        if action_dim == 0:
            axs[action_label].legend(loc='best', fontsize=8)


    axs['image'].imshow(img_strip_combined)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.subplots_adjust(hspace=0.7)
    plt.show()

    fig_lamda, ax_lamda = plt.subplots(figsize=(8, 3))
    lamda_arr = np.array(upBody_lowDim["lamda"][eps_idx])
    ax_lamda.plot(lamda_arr, label="lamda", color="orange")
    ax_lamda.set_title("Lambda Curve")
    ax_lamda.set_xlabel("Step")
    ax_lamda.set_ylabel("lamda")
    ax_lamda.legend()
    plt.show()