import sys
sys.path.append('/usr/local/lib/python2.7/site-packages')
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Sea-born Plots Configuration
# Palette
sns.set_palette("RdBu_r")
# Grid
sns.set_style("whitegrid")
# Font and Font Size
csfont = {'font': 'Helvetica',
          'size': 14}
plt.rc(csfont)


def plot_rewards(rewards, element='Pond', mean=False):
    title = 'Rewards' + ' ' + element
    if mean:
        plt.plot(rewards)
        title = 'Average-' + title
        plt.title(title)
        plt.xlabel('Epsiodes')
        plt.ylabel('Reward Values')
    else:
        plt.plot(rewards)
        plt.title(title)
        plt.xlabel('Time Steps')
        plt.ylabel('Reward Values')


def plot_height(height, element='Pond', mean=False):
    title = 'Height' + ' ' + element
    if mean:
        plt.plot(height)
        title = 'Average-' + title
        plt.title(title)
        plt.xlabel('Epsiodes')
        plt.ylabel('Meters(m)')
    else:
        for i in range(height.shape[0]):
            plt.plot(height[i, :], label=str(i))
        plt.title(title)
        plt.xlabel('Time Steps')
        plt.ylabel('Meters(m)')
        plt.legend()


def plot_flows(flows, element='Pond', mean=False):
    title = 'Flows' + ' ' + element
    if mean:
        plt.plot(flows)
        title = 'Average-' + title
        plt.title(title)
        plt.xlabel('Epsiodes')
        plt.ylabel('cu.m/sec')
    else:
        for i in range(flows.shape[0]):
            plt.plot(flows[i, :], label=str(i))
        plt.title(title)
        plt.xlabel('Time Steps')
        plt.ylabel('cu.m/sec')
        plt.legend()


def full_stack(precipitation,
               inflow,
               control_height,
               uncontrolled_height,
               flow_control,
               flow_uncontrolled,
               actions):

    plt.subplot(4, 1, 1)
    plt.plot(precipitation, label='Precipitation')
    plt.plot(inflow, label='Inflow')
    plt.title('Pond Visualization')

    plt.legend(loc='upper right', fontsize='small')

    plt.subplot(4, 1, 2)
    plt.plot(control_height, label='Uncontrolled')
    plt.plot(uncontrolled_height, linewidth=3.0, label='Control')
    plt.ylabel('Depth')
    plt.legend(loc='upper right', fontsize='small')

    plt.subplot(4, 1, 3)
    plt.plot(flow_uncontrolled, label='No Control')
    plt.plot(flow_control, label='Control')
    plt.ylabel('Outflow')

    plt.legend(loc='upper right', fontsize='small')

    plt.subplot(4, 1, 4)
    plt.plot(actions, linewidth=3.0)
    plt.ylabel('Actions')

