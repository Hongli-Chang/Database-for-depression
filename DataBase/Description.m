
%%
%raw data
 
%�����ļ��а����ĸ��׶Σ��׶�һΪʵ����С������������׶�����βʵ���²ɼ������ݣ�ʱ��Ϊ4���ӣ�������Ϊ1kHz��labelΪ0����ֹ��1�����˶����ڶ����׶�Ϊʵ����С���ǽ׶εĿ�ʵ�飬ʱ��Ϊ10���ӣ�������Ϊ1kHz��labelΪ0�������������ģ�1����δ����������ģ��׶���Ϊʵ����С�������׶�����βʵ���²ɼ������ݣ�ʱ��Ϊ4���ӣ�������Ϊ1kHz��labelΪ0����ֹ��1�����˶����׶���Ϊʵ����С������ҩ�����ú�����βʵ���²ɼ������ݣ�ʱ��Ϊ4���ӣ�������Ϊ1kHz��labelΪ0����ֹ��1�����˶���
%�����ļ��а��������׶Σ��׶ζ�Ϊ����С����ʵ����С���ڵ����׶�ͬʱ���ڿ���ʵ���²ɼ������ݣ�ʱ��Ϊ10���ӣ�������Ϊ1kHz��labelΪ0�������������ģ�1����δ����������ģ��׶���Ϊ������С����ʵ����С������׶�ͬʱ������βʵ���²ɼ������ݣ�ʱ��Ϊ4���ӣ�������Ϊ1kHz��labelΪ0����ֹ��1�����˶���
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