import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec

# 系统参数
num_particles = 300
time_steps = 1000
box_size = 20
dt = 0.1
T0 = 300
alpha = 300
center = np.array([box_size // 2, box_size // 2, 0])
r = 3
D0 = 0.1
beta = 0.05
gravity = 0.3
reaction_rate = 0.000001

# 气体参数
N2_ratio = 0.78
thermal_conductivity = 0.024
specific_heat = 1005

# 反应追踪参数
C25H52_molecules = 100  # 初始燃料分子
H2O_molecules = 0  # 初始水分子
O2_needed_per_reaction = 38  # 每个C25H52需要38个O2分子
CO2_produced_per_reaction = 25  # 每个C25H52产生25个CO2分子

def initialize_particles(num_particles, box_size, initial_concentration=1.0):
    positions = np.random.uniform(0, box_size, (num_particles, 3))
    velocities = np.zeros((num_particles, 3))
    concentrations = np.ones(num_particles) * initial_concentration
    temperatures = np.ones(num_particles) * T0
    return positions, velocities, concentrations, temperatures

def heat_conduction(temperatures, positions, thermal_conductivity, dt):
    kdT = thermal_conductivity * dt
    for i in range(len(positions)):
        distances = np.linalg.norm(positions - positions[i], axis=1)
        nearby = distances < 2.0
        if np.sum(nearby) > 1:
            temp_diff = temperatures[nearby] - temperatures[i]
            temperatures[i] += np.mean(temp_diff) * kdT
    return temperatures

def update_particles(positions, velocities, concentrations, temperatures,
                    other_concentrations, thermal_conductivity, dt, box_size, particle_type='O2'):
    # 热传导更新
    temperatures = heat_conduction(temperatures, positions, thermal_conductivity, dt)
    
    # 温度相关扩散系数
    D = D0 * (1 + beta * (temperatures - T0))
    
    # 温度梯度浮力
    temp_gradient = np.gradient(temperatures)
    buoyancy = np.zeros_like(velocities)
    buoyancy[:, 2] = gravity * (temp_gradient / T0)
    
    # 气体密度差异
    molecular_mass_ratio = 1.5 if particle_type == 'CO2' else 1.0
    density_factor = concentrations * molecular_mass_ratio
    if len(other_concentrations) > 0:
        density_factor -= np.mean(other_concentrations)
    buoyancy[:, 2] -= gravity * density_factor
    
    # 热对流横向运动
    convection_strength = 0.1 * np.abs(temp_gradient)
    buoyancy[:, 0] += np.random.normal(0, convection_strength, positions.shape[0])
    buoyancy[:, 1] += np.random.normal(0, convection_strength, positions.shape[0])
    
    # 更新运动
    brownian_motion = np.random.normal(0, np.sqrt(2 * D * dt)[:, np.newaxis], positions.shape)
    velocities += brownian_motion + buoyancy * dt
    velocities *= 0.99
    
    # 限制速度
    speed = np.linalg.norm(velocities, axis=1)
    max_speed = 2.0
    mask = speed > max_speed
    if np.any(mask):
        velocities[mask] *= max_speed / speed[mask, np.newaxis]
    
    # 更新位置
    positions += velocities * dt
    
    # 边界条件
    for i in range(3):
        bounce_mask = (positions[:, i] < 0) | (positions[:, i] > box_size)
        positions[bounce_mask, i] = np.clip(positions[bounce_mask, i], 0, box_size)
        velocities[bounce_mask, i] *= -0.5
        
    return positions, velocities, concentrations, temperatures

def update_particles_with_combustion(
    positions, velocities, concentrations, temperatures, 
    other_concentrations, thermal_conductivity, dt, box_size, reaction_rate, particle_type='O2'
):
    # 更新热传导
    temperatures = heat_conduction(temperatures, positions, thermal_conductivity, dt)
    
    # 温度相关扩散系数
    D = D0 * (1 + beta * (temperatures - T0))
    
    # 计算燃烧反应
    if particle_type == 'O2':
        fuel_concentration = other_concentrations
        combustion_rate = reaction_rate * concentrations * fuel_concentration
        combustion_rate = np.clip(combustion_rate, 0, concentrations)
        concentrations -= combustion_rate * dt
        fuel_concentration -= combustion_rate * dt
        other_concentrations = fuel_concentration
        co2_generated = combustion_rate * dt
        temperatures += co2_generated * alpha
    
    # 更新浮力
    temp_gradient = np.gradient(temperatures)
    buoyancy = np.zeros_like(velocities)
    buoyancy[:, 2] = gravity * (temp_gradient / T0)
    
    # 更新运动
    brownian_motion = np.random.normal(0, np.sqrt(2 * D * dt)[:, np.newaxis], positions.shape)
    velocities += brownian_motion + buoyancy * dt
    velocities *= 0.99

    # 限制速度
    speed = np.linalg.norm(velocities, axis=1)
    max_speed = 2.0
    mask = speed > max_speed
    velocities[mask] *= max_speed / speed[mask, np.newaxis]

    # 更新位置
    positions += velocities * dt
    
    # 边界条件
    for i in range(3):
        bounce_mask = (positions[:, i] < 0) | (positions[:, i] > box_size)
        positions[bounce_mask, i] = np.clip(positions[bounce_mask, i], 0, box_size)
        velocities[bounce_mask, i] *= -0.5
        
    return positions, velocities, concentrations, temperatures, other_concentrations

def get_concentration_at_height(positions, concentrations, height, tolerance=0.5):
    mask = np.abs(positions[:, 1] - height) < tolerance
    return np.mean(concentrations[mask]) if np.any(mask) else 0

# 初始化粒子
positions_o2, velocities_o2, concentrations_o2, temps_o2 = initialize_particles(num_particles, box_size, 0.21)
positions_co2, velocities_co2, concentrations_co2, temps_co2 = initialize_particles(num_particles, box_size, 0.0001)
positions_n2, velocities_n2, concentrations_n2, temps_n2 = initialize_particles(num_particles, box_size, N2_ratio)

# 创建图形和子图
fig = plt.figure(figsize=(16, 12))
gs = gridspec.GridSpec(2, 2)
ax_3d = fig.add_subplot(gs[:, 0], projection='3d')
ax_conc = fig.add_subplot(gs[0, 1])
ax_counts = fig.add_subplot(gs[1, 1])

# 设置3D视角
ax_3d.view_init(elev=20, azim=45)

# 初始化存储浓度历史的列表
o2_top_history = []
o2_bottom_history = []
co2_top_history = []
co2_bottom_history = []
time_history = []

def animate(t):
    global positions_o2, velocities_o2, concentrations_o2, temps_o2
    global positions_co2, velocities_co2, concentrations_co2, temps_co2
    global positions_n2, velocities_n2, concentrations_n2, temps_n2
    global o2_top_history, o2_bottom_history, co2_top_history, co2_bottom_history, time_history
    
    # 更新粒子状态
    positions_o2, velocities_o2, concentrations_o2, temps_o2, concentrations_fuel = update_particles_with_combustion(
        positions_o2, velocities_o2, concentrations_o2, temps_o2,
        concentrations_co2, thermal_conductivity, dt, box_size, reaction_rate, 'O2'
    )
    
    positions_co2, velocities_co2, concentrations_co2, temps_co2 = update_particles(
        positions_co2, velocities_co2, concentrations_co2, temps_co2,
        concentrations_o2, thermal_conductivity, dt, box_size, 'CO2'
    )
    
    positions_n2, velocities_n2, concentrations_n2, temps_n2 = update_particles(
        positions_n2, velocities_n2, concentrations_n2, temps_n2,
        np.array([]), thermal_conductivity, dt, box_size, 'N2'
    )
    
    # 清除所有子图
    for ax in plt.gcf().get_axes():
        ax.clear()

    # 3D可视化
    temps_min = min(np.min(temps_o2), np.min(temps_co2), np.min(temps_n2))
    temps_max = max(np.max(temps_o2), np.max(temps_co2), np.max(temps_n2))
    
    scatter_o2 = ax_3d.scatter(positions_o2[:, 0], positions_o2[:, 1], positions_o2[:, 2], 
                              c=temps_o2, cmap='YlOrBr', vmin=temps_min, vmax=temps_max,
                              label='O₂', alpha=0.8)
    
    scatter_co2 = ax_3d.scatter(positions_co2[:, 0], positions_co2[:, 1], positions_co2[:, 2], 
                               c=temps_co2, cmap='RdPu', vmin=temps_min, vmax=temps_max,
                               label='CO₂', alpha=0.8)
    
    scatter_n2 = ax_3d.scatter(positions_n2[:, 0], positions_n2[:, 1], positions_n2[:, 2], 
                              c=temps_n2, cmap='YlGn', vmin=temps_min, vmax=temps_max,
                              label='N₂', alpha=0.5)

    # 计算特定高度的浓度
    conc_o2_top = get_concentration_at_height(positions_o2, concentrations_o2, 17.5)
    conc_o2_bottom = get_concentration_at_height(positions_o2, concentrations_o2, 2.5)
    conc_co2_top = get_concentration_at_height(positions_co2, concentrations_co2, 17.5)
    conc_co2_bottom = get_concentration_at_height(positions_co2, concentrations_co2, 2.5)

    # 更新浓度历史
    o2_top_history.append(conc_o2_top)
    o2_bottom_history.append(conc_o2_bottom)
    co2_top_history.append(conc_co2_top)
    co2_bottom_history.append(conc_co2_bottom)
    time_history.append(t)

    # 仅保留最近的100个时间点
    history_length = 100
    if len(time_history) > history_length:
        time_history = time_history[-history_length:]
        o2_top_history = o2_top_history[-history_length:]
        o2_bottom_history = o2_bottom_history[-history_length:]
        co2_top_history = co2_top_history[-history_length:]
        co2_bottom_history = co2_bottom_history[-history_length:]

    # 绘制浓度图
    ax_conc.plot(time_history, o2_top_history, 'b-', label='O₂ (y=17.5)', alpha=0.5)
    ax_conc.plot(time_history, o2_bottom_history, 'b--', label='O₂ (y=2.5)')
    ax_conc.plot(time_history, co2_top_history, 'r-', label='CO₂ (y=17.5)', alpha=0.5)
    ax_conc.plot(time_history, co2_bottom_history, 'r--', label='CO₂ (y=2.5)')

    # 计算并绘制粒子数量
    total_o2 = np.sum(concentrations_o2)
    total_co2 = np.sum(concentrations_co2)
    
    ax_counts.bar(['O₂', 'CO₂'], [total_o2, total_co2], color=['blue', 'red'])
    ax_counts.set_ylim(0, num_particles)

    # 设置样式
    ax_3d.set_title(f"气体分布 - 时间步 {t}", color='white')
    ax_3d.set_xlabel("X", color='white')
    ax_3d.set_ylabel("Y", color='white')
    ax_3d.set_zlabel("Z", color='white')
    
    ax_conc.set_title("y=17.5和y=2.5处的气体浓度", color='white')
    ax_conc.set_xlabel("时间步", color='white')
    ax_conc.set_ylabel("浓度", color='white')
    
    ax_counts.set_title("总粒子数", color='white')
    ax_counts.set_ylabel("粒子数量", color='white')

    # 设置通用样式元素
    for ax in [ax_3d, ax_conc, ax_counts]:
        ax.set_facecolor('black')
        ax.tick_params(colors='white')
        for spine in ax.spines.values():
            spine.set_color('white')
            
    # 添加网格和图例
    ax_conc.grid(True, color='gray', alpha=0.3)
    ax_conc.legend(loc='upper right')
    ax_3d.legend(loc='upper right')

    # 设置坐标轴范围
    ax_3d.set_xlim(0, box_size)
    ax_3d.set_ylim(0, box_size)
    ax_3d.set_zlim(0, box_size)

    return scatter_o2, scatter_co2, scatter_n2
fig = plt.figure(figsize=(16, 12))
gs = gridspec.GridSpec(2, 2)
ax_3d = fig.add_subplot(gs[:, 0], projection='3d')
ax_conc = fig.add_subplot(gs[0, 1])
ax_counts = fig.add_subplot(gs[1, 1])

# Set initial view for 3D plot
ax_3d.view_init(elev=20, azim=45)

# Create animation
ani = FuncAnimation(fig, animate, frames=time_steps, interval=50, blit=False)

# Adjust layout
plt.tight_layout()
plt.show()