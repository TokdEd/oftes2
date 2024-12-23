import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec
import pandas as pd
# 系统参数
num_particles = 10000
time_steps = 1000
box_size = np.array([20, 20, 20])
dt = 1
T0 = 300
alpha = 100
center = np.array([box_size[0] // 2, box_size[1] // 2, 10])
r = 3
D0 = 0.2
beta = 0.05
gravity = 0.8
reaction_rate = 0.000001

# 气体参数
#N2_ratio = 0.78
thermal_conductivity = 0.024
specific_heat = 1005
activation_energy = 8.314 * 300
R = 8.314

# 数据记录器
class DataRecorder:
    def __init__(self):
        self.temp_history = []
        self.z_high_conc_history = []  # concentration at z=17.5
        self.z_low_conc_history = []   # concentration at z=2.5
    
    def record(self, temps_o2, temps_co2, positions, particle_types):
        try:
            # Record average temperature
            avg_temp = np.mean([np.mean(temps_o2), np.mean(temps_co2)])
            self.temp_history.append(avg_temp)
            
            # Calculate CO2 concentration at z=17.5 and z=2.5
            co2_mask = particle_types == 'CO2'
            co2_positions = positions[co2_mask]
            
           
            high_z_mask = np.abs(co2_positions[:, 2] - 17.5) < 0.5
            high_z_count = np.sum(high_z_mask)
            
            
            low_z_mask = np.abs(co2_positions[:, 2] - 2.5) < 0.5
            low_z_count = np.sum(low_z_mask)
            
            # Normalize counts to get concentration
            volume_slice = box_size[0] * box_size[1] * 1.0  # 1.0 is the slice thickness
            self.z_high_conc_history.append(high_z_count / volume_slice)
            self.z_low_conc_history.append(low_z_count / volume_slice)
            
        except Exception as e:
            print(f"Data recording error: {e}")


recorder = DataRecorder()

def initialize_particles(num_particles, box_size):
    """初始化粒子"""
    positions = np.random.uniform(0, box_size[0], (num_particles, 3))
    velocities = np.random.normal(0, 0.1, (num_particles, 3))
    temperatures = np.ones(num_particles) * T0
    particle_types = np.empty(num_particles, dtype='U3')
    return positions, velocities, temperatures, particle_types

def heat_conduction(temperatures, positions, thermal_conductivity, dt):
    """热传导计算"""
    kdT = thermal_conductivity * dt
    for i in range(len(positions)):
        distances = np.linalg.norm(positions - positions[i], axis=1)
        nearby = distances < 2.0
        if np.sum(nearby) > 1:
            temp_diff = temperatures[nearby] - temperatures[i]
            temperatures[i] += np.mean(temp_diff) * kdT
    return temperatures

def update_particles_with_combustion(positions, velocities, temperatures, particle_types,
                                   dt, box_size, reaction_rate):
    """更新粒子状态，包括化学反应和物理运动"""
    # 热传导更新
    temperatures = heat_conduction(temperatures, positions, thermal_conductivity, dt)
    
    # 扩散系数更新（考虑温度影响）
    D = D0 * (1 + beta * (temperatures - T0))
    
    # 处理化学反应
    o2_mask = particle_types == 'O2'
    co2_mask = particle_types == 'CO2'
    print(f"co2_mask: {co2_mask}")
    molecular_mass = np.ones(len(positions))
    print(f"molecular_mass: {molecular_mass}")
    print(f"Length of co2_mask: {len(co2_mask)}")
    print(f"Length of molecular_mass: {len(molecular_mass)}")
    # 假設 co2_mask 是多了一個元素
    
    if np.any(o2_mask):
        distances_to_fire = np.linalg.norm(positions[o2_mask] - center, axis=1)
        reaction_mask = distances_to_fire < r
    
        reacting_indices = np.where(o2_mask)[0][reaction_mask]
        reacting_indices = [i for i in reacting_indices if particle_types[i] != 'CO2']
        if len(reacting_indices) >= 2:
        # 隨機選擇其中一個粒子進行反應
            chosen_index = np.random.choice(reacting_indices)
        # 更新選中粒子的屬性
            particle_types[chosen_index] = 'CO2'
            temperatures[chosen_index] += alpha
            positions[chosen_index] = center + np.random.normal(0, 0.1, (1, 3))
            velocities[chosen_index] = np.random.normal(0, 0.1, (1, 3))
        
        # 找到另一個粒子的索引
            other_index = [i for i in reacting_indices if i != chosen_index][0]
        
        # 確保刪除粒子後，所有數組大小一致
        # 刪除另一個粒子
            particle_types = np.delete(particle_types, other_index, axis=0)
            positions = np.delete(positions, other_index, axis=0)
            velocities = np.delete(velocities, other_index, axis=0)
            temperatures = np.delete(temperatures, other_index, axis=0)
            #assert len(co2_mask) == len(molecular_mass), f"co2_mask size: {len(co2_mask)}, molecular_mass size: {len(molecular_mass)}"
        # 同步更新 co2_mask 和 molecular_mass
            co2_mask = np.delete(co2_mask, other_index, axis=0)
            molecular_mass = np.delete(molecular_mass, other_index, axis=0)
            assert len(co2_mask) == len(molecular_mass), f"co2_mask size: {len(co2_mask)}, molecular_mass size: {len(molecular_mass)}"
                # 温度梯度引起的对流
    temp_gradient = np.gradient(temperatures)
    convection = np.zeros_like(velocities)
    
    # 垂直方向的温度梯度对流
    convection[:, 2] = 0.2 * temp_gradient
    
    # 复杂涡流效应
    distance_to_center = np.linalg.norm(positions[:, :2] - center[:2], axis=1)
    angle = np.arctan2(positions[:, 1] - center[1], positions[:, 0] - center[0])
    
    # 温度影响涡流强度
    temp_factor = (temperatures - T0) / T0
    vortex_strength = 0.15 * np.exp(-distance_to_center / 8.0) * (1 + temp_factor)
    
    # 添加螺旋上升运动
    height_factor = np.exp(-(positions[:, 2] - center[2]) / 10.0)
    convection[:, 0] += vortex_strength * np.sin(angle) * height_factor
    convection[:, 1] -= vortex_strength * np.cos(angle) * height_factor
    convection[:, 2] += 0.1 * vortex_strength * height_factor

    # 气体密度差异导致的浮力
    molecular_mass = np.ones_like(temperatures)
    molecular_mass[co2_mask] = 1.5  # CO2更重
    buoyancy = np.zeros_like(velocities)
    buoyancy[:, 2] = gravity * (1.0 - molecular_mass * (T0 / temperatures))

    # 布朗运动
    D = D0 * (1 + beta * (temperatures - T0))
    D = D[:, np.newaxis]

    """
    #brownian_strength = 1.0
    #brownian_motion = np.random.normal(0, np.sqrt(2 * D * dt) * brownian_strength, positions.shape)

    """
    #brownian_strength = 1.0
    #brownian_motion = np.random.normal(0, np.sqrt(2 * D * dt) * brownian_strength, positions.shape)
   # 合并所有力的作用（移除了分子间作用力）
    # 如果 brownian_strength 是數組
    brownian_strength = 1.0
    brownian_strength = np.full(positions.shape[0], brownian_strength)
    brownian_motion = np.random.normal(0, np.sqrt(2 * D * dt) * brownian_strength[:, np.newaxis], positions.shape)
    total_acceleration = (convection + buoyancy) / molecular_mass[:, np.newaxis]
    
    # 更新速度（考虑空气阻力）
    drag_coef = 0.1
    air_resistance = -drag_coef * velocities * np.linalg.norm(velocities, axis=1)[:, np.newaxis]
    velocities += (total_acceleration + air_resistance) * dt + brownian_motion
    
    # 能量损耗
    velocities *= 0.99

    # 速度限制
    speed = np.linalg.norm(velocities, axis=1)
    max_speed = 3.0  # 增加最大速度限制
    mask = speed > max_speed
    if np.any(mask):
        velocities[mask] *= max_speed / speed[mask, np.newaxis]

    # 更新位置
    positions += velocities * dt

    # 边界条件（弹性碰撞）
    for i in range(3):
        bounce_mask = (positions[:, i] < 0) | (positions[:, i] > box_size[i])
        positions[bounce_mask, i] = np.clip(positions[bounce_mask, i], 0, box_size[i])
        velocities[bounce_mask, i] *= -0.8  # 非完全弹性碰撞

    return positions, velocities, temperatures, particle_types

# 初始化粒子
num_particles_o2 = int(num_particles * 0.21)
num_particles_co2 = int(num_particles * 0.01)


positions_o2, velocities_o2, temps_o2, types_o2 = initialize_particles(num_particles_o2, box_size)
positions_co2, velocities_co2, temps_co2, types_co2 = initialize_particles(num_particles_co2, box_size)

# 设置粒子类型
types_o2[:] = 'O2'
types_co2[:] = 'CO2'


# 合并所有粒子数组
positions = np.vstack([positions_o2, positions_co2])
velocities = np.vstack([velocities_o2, velocities_co2])
temperatures = np.concatenate([temps_o2, temps_co2])
particle_types = np.concatenate([types_o2, types_co2])

# 创建图形
fig = plt.figure(figsize=(15, 10))
gs = gridspec.GridSpec(2, 2)
ax = fig.add_subplot(gs[:, 0], projection='3d')
ax_temp = fig.add_subplot(gs[0, 1])
ax_conc = fig.add_subplot(gs[1, 1])

# Set black background
fig.patch.set_facecolor('black')
for ax_i in [ax, ax_temp, ax_conc]:
    ax_i.set_facecolor('black')
    ax_i.tick_params(colors='white')
    ax_i.xaxis.label.set_color('white')
    ax_i.yaxis.label.set_color('white')
    if hasattr(ax_i, 'zaxis'):
        ax_i.zaxis.label.set_color('white')

def calculate_co2_density(positions, particle_types, y_plane, box_size, resolution=20):
    """Calculate CO2 density at a specific y-plane"""
    co2_mask = particle_types == 'CO2'
    co2_positions = positions[co2_mask]
    
    # Filter particles near the y-plane (within ±0.5 units)
    near_plane = np.abs(co2_positions[:, 1] - y_plane) < 0.5
    relevant_positions = co2_positions[near_plane]
    
    # Create 2D histogram
    x_edges = np.linspace(0, box_size[0], resolution)
    z_edges = np.linspace(0, box_size[2], resolution)
    
    density, _, _ = np.histogram2d(
        relevant_positions[:, 0] if len(relevant_positions) > 0 else [],
        relevant_positions[:, 2] if len(relevant_positions) > 0 else [],
        bins=[x_edges, z_edges]
    )
    
    return density.T
def plot_co2_density(density, z_plane, box_size, filename):
    """繪製並儲存特定z截面的CO2密度圖"""
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.imshow(density, extent=[0, box_size[0], 0, box_size[1]], origin='lower', cmap='viridis')
    ax.set_title(f'CO2 Density at z={z_plane}')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    fig.colorbar(cax, ax=ax, label='Density')
    plt.savefig(filename)
    plt.close(fig)
def save_concentration_data(recorder, filename='concentration_data.csv'):
    """Save concentration data to CSV file"""
    # Create a DataFrame with the concentration data
    df = pd.DataFrame({
        'Time_Step': range(len(recorder.z_high_conc_history)),
        'Z17.5_Concentration': recorder.z_high_conc_history,
        'Z2.5_Concentration': recorder.z_low_conc_history
    })
    
    # Save to CSV
    df.to_csv(filename, index=False)
    print(f"Concentration data saved to {filename}")

def save_concentration_plot(recorder, filename='concentration_plot.png'):
    """Create and save a high-quality concentration plot"""
    # Create a new figure with larger size and higher DPI
    plt.figure(figsize=(12, 8), dpi=300)
    
    # Set dark background
    plt.gca().set_facecolor('black')
    plt.gcf().set_facecolor('black')
    
    # Plot concentration data
    time_steps = range(len(recorder.z_high_conc_history))
    plt.plot(time_steps, recorder.z_high_conc_history, 'r-', 
             label='z=17.5', linewidth=2)
    plt.plot(time_steps, recorder.z_low_conc_history, 'b-', 
             label='z=2.5', linewidth=2)
    
    # Customize plot
    plt.title("CO₂ Concentration at Different Heights", color='white', size=14)
    plt.xlabel("Time Step", color='white', size=12)
    plt.ylabel("Concentration", color='white', size=12)
    plt.grid(True, color='gray', alpha=0.3)
    
    # Customize ticks
    plt.tick_params(colors='white')
    
    # Add legend with white text
    legend = plt.legend(loc='upper right', facecolor='black', edgecolor='white')
    for text in legend.get_texts():
        text.set_color('white')
    
    # Save plot with tight layout
    plt.tight_layout()
    plt.savefig(filename, facecolor='black', edgecolor='none', bbox_inches='tight')
    plt.close()
    print(f"Concentration plot saved to {filename}")
def animate(frame):
    global positions, velocities, temperatures, particle_types
    
    # Update particle states
    positions, velocities, temperatures, particle_types = update_particles_with_combustion(
        positions, velocities, temperatures, particle_types,
        dt, box_size, reaction_rate
    )
    
    # Clear old plots
    ax.clear()
    ax_temp.clear()
    ax_conc.clear()
    
    ax.set_facecolor('black')
    
    # Plot particles in 3D
    o2_mask = particle_types == 'O2'
    co2_mask = particle_types == 'CO2'
    
    temps_min = min(np.min(temperatures), T0)
    temps_max = max(np.max(temperatures), T0 + alpha)
    
    ax.scatter(positions[o2_mask, 0], positions[o2_mask, 1], positions[o2_mask, 2],
              c=temperatures[o2_mask], cmap='YlOrBr', vmin=temps_min, vmax=temps_max,
              label='O₂', alpha=0.8)
    ax.scatter(positions[co2_mask, 0], positions[co2_mask, 1], positions[co2_mask, 2],
              c=temperatures[co2_mask], cmap='RdPu', vmin=temps_min, vmax=temps_max,
              label='CO₂', alpha=0.8)
    
    # Plot heat source
    ax.scatter(center[0], center[1], center[2],
              color='orange', s=200, label='Heat Source',
              edgecolor='red', linewidth=2)
    
    # Set 3D axes
    ax.set_xlim([0, box_size[0]])
    ax.set_ylim([0, box_size[1]])
    ax.set_zlim([0, box_size[2]])
    ax.set_xlabel('X', color='white')
    ax.set_ylabel('Y', color='white')
    ax.set_zlabel('Z', color='white')
    
    # Set label colors
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.tick_params(axis='z', colors='white')
    
    # Add legend to 3D plot
    legend = ax.legend(loc='upper right', facecolor='black', edgecolor='white')
    for text in legend.get_texts():
        text.set_color('white')
    
    # Add grid to 3D plot
    ax.grid(True, color='gray', alpha=0.3)
    
    # Plot temperature history
    if len(recorder.temp_history) > 0:
        ax_temp.plot(recorder.temp_history, color='white')
    ax_temp.set_title("Average Temperature History", color='white')
    ax_temp.grid(True, color='gray', alpha=0.3)
    ax_temp.set_ylabel("Temperature (K)", color='white')
    ax_temp.set_xlabel("Time Step", color='white')
    
    # Plot concentration history for both z-planes
    if len(recorder.z_high_conc_history) > 0:
        time_steps = range(len(recorder.z_high_conc_history))
        ax_conc.plot(time_steps, recorder.z_high_conc_history, 'r-', 
                    label='z=2.5', linewidth=2)
        ax_conc.plot(time_steps, recorder.z_low_conc_history, 'b-', 
                    label='z=17.5', linewidth=2)
        
    ax_conc.set_title("CO₂ Concentration at Different Heights", color='white')
    ax_conc.set_xlabel("Time Step", color='white')
    ax_conc.set_ylabel("Concentration", color='white')
    ax_conc.grid(True, color='gray', alpha=0.3)
    
    # Add legend to concentration plot with white text
    legend = ax_conc.legend(loc='upper right', facecolor='black', edgecolor='white')
    for text in legend.get_texts():
        text.set_color('white')
    
    # Record data
    try:
        o2_temps = temperatures[o2_mask]
        co2_temps = temperatures[co2_mask]
        
        if len(o2_temps) > 0 and len(co2_temps) > 0:
            recorder.record(o2_temps, co2_temps, positions, particle_types)
    except Exception as e:
        print(f"Data recording error: {e}")

def save_simulation_data(recorder):
    """Save both the data and plot after simulation"""
    save_concentration_data(recorder)
    save_concentration_plot(recorder)
# Create animation
ani = FuncAnimation(fig, animate, frames=time_steps, interval=50, blit=False)

# Adjust layout and display
plt.tight_layout()
plt.show()
save_simulation_data(recorder)
ani.save('simulation_animation.mp4', writer='ffmpeg', fps=60, dpi=150)
