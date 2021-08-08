
%%
%raw data
 
%抑郁文件夹包含四个阶段：阶段一为实验组小鼠最初的正常阶段在悬尾实验下采集的数据，时长为4分钟，采样率为1kHz，label为0代表静止，1代表运动；第二个阶段为实验组小鼠焦虑阶段的矿场实验，时长为10分钟，采样率为1kHz，label为0代表进入旷场中心，1代表未进入旷场中心；阶段三为实验组小鼠抑郁阶段在悬尾实验下采集的数据，时长为4分钟，采样率为1kHz，label为0代表静止，1代表运动；阶段四为实验组小鼠抑郁药物作用后在悬尾实验下采集的数据，时长为4分钟，采样率为1kHz，label为0代表静止，1代表运动。
%对照文件夹包含二个阶段：阶段二为正常小鼠与实验组小鼠在第三阶段同时期在旷场实验下采集的数据，时长为10分钟，采样率为1kHz，label为0代表进入旷场中心，1代表未进入旷场中心；阶段三为对照组小鼠与实验组小鼠第三阶段同时期在悬尾实验下采集的数据，时长为4分钟，采样率为1kHz，label为0代表静止，1代表运动。
% The Depression folder contains four phases: phase01 is the data collected under the tail suspension test in the initial normal stage of the experimental group of mice, the duration is 4 minutes, the sampling rate is 1kHz, the label is 0 for static and 1 for exercise;
% phase02 is the open field test in the anxiety phase of the experimental group of mice, with a duration of 10 minutes and a sampling rate of 1 kHz, the label of 0 means entering the center of the open field, and 1 means not entering the center of the open field;
% phase03 is the data collected under the tail suspension test in the depression stage of the experimental group of mice, the duration is 4 minutes, the sampling rate is 1kHz, the label is 0 for static and 1 for exercise;
% phase04 is the data collected in the tail suspension test after the depressive drugs in the experimental group of mice, the duration is 4 minutes, the sampling rate is 1kHz, and the label is 0 for static and 1 for exercise.
% The Control folder contains two phases: phase02 is the data collected by normal mice in the open field test, and the collection time is the same as that of the experimental group mice in the third stage.
% The duration is 10 minutes, the sampling rate is 1kHz, label 0 means entering the center of the open field, and 1 means not entering the center of the open field;
% phase03 is the data collected in the tail suspension experiment during the third stage of the control mice and the experimental group mice at the same time. The duration is 4 minutes, the sampling rate is 1kHz, and the label is 0 for static and 1 for exercise.

%%
%feature data
%The extracted feature data corresponds to the original data. The original data is cut into one segment per second, and the features of five frequency bands are extracted, and the label becomes one label per second.

%%
%expriment data
The experimental data corresponds to the six sets of experiments and two algorithms in the paper