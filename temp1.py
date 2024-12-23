import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import matplotlib.gridspec as gridspec

# 系统参数
num_particles = 3000
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
        self.concentration_history = []
        self.energy_history = []
    
    def record(self, temps_o2, temps_co2, 
               conc_o2, conc_co2, 
               velocities_o2, velocities_co2):
        try:
            # 记录平均温度
            avg_temp = np.mean([np.mean(temps_o2), np.mean(temps_co2)])
            self.temp_history.append(avg_temp)
            
            # 记录平均浓度
            avg_conc_o2 = np.mean(conc_o2) if len(conc_o2) > 0 else 0
            avg_conc_co2 = np.mean(conc_co2) if len(conc_co2) > 0 else 0
            self.concentration_history.append([avg_conc_o2, avg_conc_co2])
            
            # 记录系统总能量
            total_ke = (np.sum(np.linalg.norm(velocities_o2, axis=1)**2) +
                       np.sum(np.linalg.norm(velocities_co2, axis=1)**2) ) / 2
            total_thermal = (np.sum(temps_o2) + np.sum(temps_co2) ) * specific_heat
            self.energy_history.append(total_ke + total_thermal)
        except Exception as e:
            print(f"数据记录错误: {e}")

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
    
    if np.any(o2_mask):
        distances_to_fire = np.linalg.norm(positions[o2_mask] - center, axis=1)
        reaction_mask = distances_to_fire < r
        
        reacting_indices = np.where(o2_mask)[0][reaction_mask]
        if len(reacting_indices) > 0:
            particle_types[reacting_indices] = 'CO2'
            temperatures[reacting_indices] += alpha
            positions[reacting_indices] = center + np.random.normal(0, 0.1, (len(reacting_indices), 3))
            velocities[reacting_indices] = np.random.normal(0, 0.1, (len(reacting_indices), 3))

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
    D = D[:, np.newaxis]
    brownian_strength = 1.0
    brownian_motion = np.random.normal(0, np.sqrt(2 * D * dt) * brownian_strength, positions.shape)

    # 合并所有力的作用（移除了分子间作用力）
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

# 设置黑色背景
fig.patch.set_facecolor('black')
for ax_i in [ax, ax_temp, ax_conc]:
    ax_i.set_facecolor('black')
    ax_i.tick_params(colors='white')
    ax_i.xaxis.label.set_color('white')
    ax_i.yaxis.label.set_color('white')
    if hasattr(ax_i, 'zaxis'):
        ax_i.zaxis.label.set_color('white')

def animate(frame):
    global positions, velocities, temperatures, particle_types
    
    # 更新粒子状态
    positions, velocities, temperatures, particle_types = update_particles_with_combustion(
        positions, velocities, temperatures, particle_types,
        dt, box_size, reaction_rate
    )
    
    # 清除旧图
    ax.clear()
    ax.set_facecolor('black')
    
    # 绘制粒子
    o2_mask = particle_types == 'O2'
    co2_mask = particle_types == 'CO2'

    
    # 创建自定义colormap
    temps_min = min(np.min(temperatures), T0)
    temps_max = max(np.max(temperatures), T0 + alpha)
    
    # 绘制粒子
    ax.scatter(positions[o2_mask, 0], positions[o2_mask, 1], positions[o2_mask, 2],
              c=temperatures[o2_mask], cmap='YlOrBr', vmin=temps_min, vmax=temps_max,
              label='O₂', alpha=0.8)
    ax.scatter(positions[co2_mask, 0], positions[co2_mask, 1], positions[co2_mask, 2],
              c=temperatures[co2_mask], cmap='RdPu', vmin=temps_min, vmax=temps_max,
              label='CO₂', alpha=0.8)

    
    # 绘制热源
    ax.scatter(center[0], center[1], center[2],
              color='orange', s=200, label='Heat Source',
              edgecolor='red', linewidth=2)
    
    # 设置坐标轴
    ax.set_xlim([0, box_size[0]])
    ax.set_ylim([0, box_size[1]])
    ax.set_zlim([0, box_size[2]])
    ax.set_xlabel('X', color='white')
    ax.set_ylabel('Y', color='white')
    ax.set_zlabel('Z', color='white')
    
    # 设置标签颜色
    ax.tick_params(axis='x', colors='white')
    ax.tick_params(axis='y', colors='white')
    ax.tick_params(axis='z', colors='white')
    
    # 添加图例
    legend = ax.legend(loc='upper right', facecolor='black', edgecolor='white')
    for text in legend.get_texts():
        text.set_color('white')
    
    # 添加网格
    ax.grid(True, color='gray', alpha=0.3)
    
    # 更新温度历史
    if len(recorder.temp_history) > 0:
        ax_temp.plot(recorder.temp_history, color='white')
    ax_temp.set_title("Average Temperature History", color='white')
    ax_temp.grid(True, color='gray', alpha=0.3)
    
    # 更新浓度历史
    if len(recorder.concentration_history) > 0:
        concentration_history = np.array(recorder.concentration_history)
        ax_conc.plot(concentration_history[:, 0], 'b-', label='O₂')  # 添加标签
        ax_conc.plot(concentration_history[:, 1], 'r-', label='CO₂')  # 添加标签
    ax_conc.set_title("Gas Concentration History", color='white')
    ax_conc.grid(True, color='gray', alpha=0.3)
    if len(recorder.concentration_history) > 0:  # 只在有数据时添加图例
        ax_conc.legend()
    
    # 记录数据
    try:
        o2_temps = temperatures[o2_mask]
        co2_temps = temperatures[co2_mask]
        
        if len(o2_temps) > 0 and len(co2_temps) > 0 :
            recorder.record(o2_temps, co2_temps, 
                          np.ones_like(o2_temps), np.ones_like(co2_temps),
                          velocities[o2_mask], velocities[co2_mask])
    except Exception as e:
        print(f"数据记录错误: {e}")

# 修改动画创建
ani = FuncAnimation(fig, animate, frames=time_steps, interval=50, blit= False)
#anim = FuncAnimation(fig, animate, frames=range(time_steps), interval=dt * 1000, repeat=False)

# 保存动画为 mp4 文件
# 调整布局并显示
plt.tight_layout()
plt.show()
ani.save('simulation_animation.mp4', writer='ffmpeg', fps=60,dpi=150)