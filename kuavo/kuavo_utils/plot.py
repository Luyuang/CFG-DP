def draw_predicted_result(task_name, all_img, upBody_lowDim, eps_idx, slice_idx = None):
    import numpy as np
    import matplotlib.pyplot as plt
    img_strips = []
    show_img_num = 18
    for img_name, img_eps in all_img.items():
        img_strip = np.concatenate(np.array(img_eps[eps_idx])[::len(img_eps[eps_idx])//show_img_num], axis=1)  # Row for images
        img_strips.append(img_strip)

    img_strip_combined = np.vstack(img_strips)
    
    JOINT_DIM_LABELS= ["zarm_l1_link", "zarm_l2_link", "zarm_l3_link", "zarm_l4_link", "zarm_l5_link", "zarm_l6_link", "zarm_l7_link", 
                  "dex_hand_l1_link", "dex_hand_l2_link", "dex_hand_l3_link", "dex_hand_l4_link", "dex_hand_l5_link", "dex_hand_l6_link", 
                  "zarm_r1_link", "zarm_r2_link", "zarm_r3_link", "zarm_r4_link", "zarm_r5_link", "zarm_r6_link", "zarm_r7_link", 
                  "dex_hand_r1_link", "dex_hand_r2_link", "dex_hand_r3_link", "dex_hand_r4_link", "dex_hand_r5_link", "dex_hand_r6_link", 
                  ] 
    img_rows = len(all_img)
    figure_layout = [
            JOINT_DIM_LABELS[:7],
            JOINT_DIM_LABELS[7:13] + ['extra1'],
            JOINT_DIM_LABELS[13:-6] ,
            JOINT_DIM_LABELS[-6:]+ ['extra2'],
        ]
    for i in range(img_rows):
        row_images = ['image'] * len(JOINT_DIM_LABELS[:7])
        figure_layout.insert(i, row_images)


    plt.rcParams.update({'font.size': 7})
    fig, axs = plt.subplot_mosaic(figure_layout)
    fig.set_size_inches([30, 15]) 
    fig.suptitle(task_name, fontsize=15)
    
    if slice_idx is not None:
        SELECTED_JOINT = [x for slc in slice_idx for x in JOINT_DIM_LABELS[slc[0]:slc[1]]]
    else:
        SELECTED_JOINT = JOINT_DIM_LABELS[0:8] + JOINT_DIM_LABELS[13:21] 
    for action_dim, action_label in enumerate(SELECTED_JOINT):
        for low_dim_name, low_dim_values in upBody_lowDim.items():
            if 'pred' in low_dim_name:  
                # selected_range = range(len(low_dim_values[eps_idx]) - 8 -50)  # len(low_dim_values[eps_idx]) - 8
                # draw_pred_num = 2
                # for j in selected_range:  # select range
                #     x_range = np.arange(j, j + min(draw_pred_num, 8))  
                #     y_values = low_dim_values[eps_idx][j, :draw_pred_num, action_dim]  
                #     axs[action_label].plot(x_range, y_values,  alpha=0.5, zorder=1)
                axs[action_label].plot(low_dim_values[eps_idx][:, 0, action_dim], label=low_dim_name, alpha=1, zorder=1)

            elif 'true_actions' in low_dim_name:    # (n, 26)
                axs[action_label].plot(low_dim_values[eps_idx][:, action_dim], label=low_dim_name, alpha=0.5, zorder=1)
                
            elif 'true_states' in low_dim_name:
                axs[action_label].plot(low_dim_values[eps_idx][:, action_dim], label=low_dim_name, alpha=0.2, zorder=1)
        axs[action_label].set_xlabel(action_label, labelpad=5)  
        # axs[action_label].set_xlabel('Time in one episode')
        axs[action_label].legend()

    axs['image'].imshow(img_strip_combined)
    # axs['image'].set_xlabel('Time in one episode (subsampled)')
    # axs['image'].set_title('Image Comparison')

    plt.legend()
    plt.show()
