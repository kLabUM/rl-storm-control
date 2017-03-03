clc
clear
close all
%% Reward Functions
load rewards1

figure(1)
plot(rewards1)
xlabel('Episodes')
ylabel('Reward Value')
%% Flooding and Flows 
load flooding_1
load flooding_2
load outflow_mean
load outflow_tracker

subplot(1,3,1)
plot(flooding_2)
ylabel('Flow cu.m/sec')
xlabel('Episodes')
subplot(1,3,2)
plot(flooding_1)
xlabel('Episodes')
ylabel('Flow cu.m/sec')
subplot(1,3,3)
plot(outflow_mean)
xlabel('Episodes')
ylabel('Flow cu.m/sec')