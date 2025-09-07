import carla
from Start_server import start_server, kill
import time
from global_route_planner import GlobalRoutePlanner
from CarlaTrafficEnv import Carla_Traffic_Env
from Vision_Policy import Policy
import numpy as np
from Transformer_Memory import Memory
import time
import math
from Start_server import start_server, kill
from PyQt6.QtWidgets import QApplication
from DynamicPlotter import DynamicPlotter
from scipy.interpolate import CubicSpline
from scipy.stats import gaussian_kde
import sys
from scipy.spatial import KDTree

def get_route_paper(grp,map):

    def delete_doubles(route):
        last_loc = [(route[0])[0].transform.location.x, (route[0])[0].transform.location.y]
        unique_route = [(route[0])[0]]
        for wp,_ in route[1:]:
            loc = [wp.transform.location.x, wp.transform.location.y]
            if np.linalg.norm(np.array(loc) - np.array(last_loc)) > .3:
                unique_route.append(wp)
                last_loc = loc
        return unique_route

    # Beispiel: Start-Location selbst gemessen oder geschätzt
    start_loc = carla.Location(x=92.6, y=104.8, z=0.5)
    end_loc = carla.Location(x=87.5, y=114.9, z=0.5)

    # Nächste Waypoints auf der Straße finden
    start_wp = map.get_waypoint(start_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
    end_wp = map.get_waypoint(end_loc, project_to_road=True, lane_type=carla.LaneType.Driving)

    route = grp.trace_route(start_wp.transform.location, end_wp.transform.location)
    route = delete_doubles(route)

    x, y, z = route[0].transform.location.x, route[0].transform.location.y, route[0].transform.location.z
    spawn_point = carla.Transform(carla.Location(x=x, y=y, z=z + .5), route[0].transform.rotation)

    coords = np.array([[wp.transform.location.x, wp.transform.location.y] for wp in route])
    tree = KDTree(coords)
    return (spawn_point, route), tree

kill()
start_server(2000)


ckpt = 'checkpoints/policy_and_optimizer.pth'

policy = Policy(512,2, 0.05, 3, num_quantiles=32).cuda()
policy.load_policy(ckpt)

time.sleep(20)
env = Carla_Traffic_Env(map='Town01_Opt', CNN = policy.model.image_encoder)

grp = env.grp#GlobalRoutePlanner(env.map, sampling_resolution=2)
route, tree = get_route_paper(grp, env.map)
env.set_route(route, tree)

app = QApplication(sys.argv)
plotter = DynamicPlotter()
plotter.show()

def gmm_pdf(means, stds, n_points=200):
    x = np.linspace(-3, 3, n_points)

    # Compute densities
    steering_pdf =  1 / (math.sqrt(2 * math.pi) * stds[0]) * np.exp(-((x - means[0]) ** 2) / (2 * stds[0] ** 2))
    acceleration_pdf = 1 / (math.sqrt(2 * math.pi) * stds[1]) * np.exp(-((x - means[1]) ** 2) / (2 * stds[1] ** 2))

    return steering_pdf, acceleration_pdf


def critic_density(quantiles):
    N = 32
    p_coarse = np.linspace(1 / (N+1), N/(N+1), N)
    x_coarse = quantiles  # replace with your own quantiles if needed

    # === Step 2: Fit interpolating spline to quantiles ===
    quantile_spline = CubicSpline(p_coarse, x_coarse, bc_type='natural')

    # === Step 3: Sample many new points from interpolated quantile function ===
    M = 100

    p_dense = np.random.uniform(0, 1, size=M)
    x_dense = quantile_spline(p_dense)

    # === Step 4: KDE from interpolated samples ===

    kde = gaussian_kde(x_dense)

    x_plot = np.linspace(-6, 2.5, 200)

    pdf_est = kde(x_plot)
    return pdf_est



rmse_list = []
mean_error_list = []
speed_list = []
max_error_list = []
time_list = []
Hz = 10
max_timesteps = 60*Hz
throttles = []
steerings = []
memory = Memory()

for run, max_speed in enumerate([(30+10*i)/3.6 for i in range(7)]):
    rmse = 0
    speed = 0
    mean_error = 0
    max_error = 0
    total_reward = 0
    print(f'Max_speed: {int(max_speed*3.6)} km/h _________________________________________________')

    state, all_stats = env.reset(max_speed=max_speed)
    plotter.reset_plot4()
    plotter.v_max = max_speed*3.6
    plotter.add_point_plot4(all_stats[0, 2])
    for i in range(3):
        env.world.tick()
    dists = all_stats[:, 1]
    velocities = all_stats[:, 2]
    speed += velocities * 3.6
    rmse += dists ** 2
    mean_error += dists
    max_error = max(max_error, dists)
    all_means = 0
    fixed_message = 0
    for t in range(max_timesteps):
        actions, state_values, means, stds = policy.act(state, memory, det=True, log=True)
        actions = actions.cpu().numpy().squeeze()

        # Add messages to state
        state, reward, done, all_stats, _ = env.step(np.clip(actions / 3, -1, 1))
        steering_pdf, acceleration_pdf = gmm_pdf(2*means[0].detach().cpu().float().numpy()+np.array([state[0,0].item()*3, state[0,1].item()*3]), stds[0].detach().cpu().float().numpy())
        critic_pdf = critic_density(state_values[0].detach().cpu().float().numpy())

        plotter.set_array_plot1(steering_pdf)
        plotter.set_array_plot2(acceleration_pdf)
        plotter.set_array_plot3(critic_pdf)
        plotter.set_scatter_points_plot3(state_values[0].detach().cpu().float().numpy().tolist())
        plotter.add_point_plot4(all_stats[0, 2]*3.6)

        plotter.update_plots()
        app.processEvents()

        total_reward += reward
        dists = all_stats[:, 1]
        velocities = all_stats[:, 2]
        speed += velocities * 3.6
        rmse += dists ** 2
        mean_error += dists
        max_error = max(max_error, dists[0])
        if done or t == max_timesteps-1:
            break

    memory.rmse += np.sqrt(rmse / (t + 1))
    memory.total_rewards.append(np.round(float(total_reward)))
    memory.mean_error += mean_error / (t + 1)
    memory.speed += speed / (t + 1)

    rmse_list.append(round(float(np.sqrt(rmse / (t + 1))[0]), ndigits=3))
    time_list.append((t+1)/Hz)
    mean_error_list.append(round(float(mean_error[0]) / (t + 1), ndigits=3))
    speed_list.append(round(float(speed[0]) / (t + 1), ndigits=3))
    max_error_list.append(round(float(max_error), ndigits=3))
    print(f'Mean Cumulated Reward up to run {run}: {np.mean(np.array(memory.total_rewards))}')
    print(f'RMSE up to run {run}: {np.mean(memory.rmse/(run+1))}')
    print(f'Mean Error up to run {run}: {np.mean(memory.mean_error/(run+1))}')
    print(f'Mean Speed up to run {run}: {np.mean(memory.speed/(run+1))}')
    
print(f'Cumulated Rewards: {memory.total_rewards}')
print(f'RMSE: {rmse_list}')
print(f'Mean Error: {mean_error_list}')
print(f'Max Error: {max_error_list}')
print(f'Mean Speed: {speed_list}')
print(f'Time: {time_list}')
sys.exit(app.exec())
