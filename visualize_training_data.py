import pickle
import matplotlib.pyplot as plt
import numpy as np



def moving_average(y_values, n):
    """Computes a moving average with window size n, keeping the output size the same."""
    if not y_values or n <= 0:
        return np.array([])

    y_values = np.array(y_values)

    # Pad the edges using reflection or edge replication
    pad_size = n // 2  # Half of the window size
    padded_values = np.pad(y_values, pad_size, mode='edge')  # 'edge' or 'reflect'

    # Apply convolution
    smoothed = np.convolve(padded_values, np.ones(n) / n, mode='valid')

    return smoothed[:-1]

def generate_plots():
    with open("checkpoints/policy_and_optimizer_data.pkl", "rb") as f:

        train_history = pickle.load(f)


    fig1, ax1 = plt.subplots(1,1,figsize=(7,5))
    #fig1.canvas.manager.window.wm_geometry("+1920+0")




    reward_list = train_history["Reward"]
    filter = [isinstance(x, float) for x in reward_list]
    x = [i*4 for i in range(len(reward_list))]
    
    ax1[0].plot(x, reward_list, linewidth=2, alpha=0.5)
    ax1[0].plot(x,moving_average(reward_list,20), linewidth=2, alpha=1)
    ax1[0].set_xlabel("Episodes", fontsize=14)
    ax1[0].set_ylabel("Cumulated Reward", fontsize=14)

    plt.savefig("Reward_Collision.png")
    plt.close()

    fig2, ax2 = plt.subplots(2,2, figsize=(12, 7))

    rmse_list =train_history["RMSE"]
    

    mean_error_list = train_history["Mean Error"]
    

    speed_list = train_history["Speed"]
    

    mean_std_list = np.array(train_history["Stds"])

    ax2[0,0].plot(x, np.log(rmse_list), linewidth=1.5, alpha=0.5)
    ax2[0,0].plot(x, np.log(moving_average(rmse_list,20)), linewidth=2.5, alpha=1)
    ax2[0,0].set_ylabel("RMSE", fontsize=12)

    ax2[0,1].plot(x, np.log(mean_error_list), linewidth=1.5, alpha=0.5)
    ax2[0,1].plot(x, np.log(moving_average(mean_error_list,20)), linewidth=2.5, alpha=1)
    ax2[0,1].set_ylabel("Mean Error", fontsize=12)

    ax2[1,0].plot(x, speed_list, linewidth=1.5, alpha=0.5)
    ax2[1,0].plot(x, moving_average(speed_list,20), linewidth=2.5, alpha=1)
    ax2[1,0].set_ylabel("Avg Speed", fontsize=12)

    ax2[1,1].plot(x, mean_std_list[:,0],'g', linewidth=1.5, alpha=0.5, label='Steering')
    ax2[1,1].plot(x, moving_average(mean_std_list[:,0].tolist(),20),'g', linewidth=2.5, alpha=1)

    ax2[1,1].plot(x, mean_std_list[:,1],'b', linewidth=1.5, alpha=0.5, label='Acceleration')
    ax2[1,1].plot(x, moving_average(mean_std_list[:,1].tolist(),20),'b', linewidth=2.5, alpha=1)

    ax2[1,1].set_ylabel("Mean Standard Deviation", fontsize=12)
    ax2[1,1].legend()

    plt.savefig("AdditionalData.png")
    plt.close()
