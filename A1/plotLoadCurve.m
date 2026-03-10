%% Matlab script to plot demand data
% This script plots demand data. Due to data limitations the time horizon
% should be equal or less than 48.
% Example:
% To plot demand curve for 24 time periods we do the following:
% plotLoadCurve(24)
% W. Bukhsh, April 2014
% wbukhsh@gmail.com


function plotLoadCurve(nT)
if nT>48
    error('Due to limitations of the demand data, the value of Time period must not exceed 48')
end

data=[0.6642   0.6645    0.6517    0.6293    0.6144    0.6074    0.5935    0.5791    0.5637    0.5566    0.5553    0.5612    0.5850    0.5993    0.6225    0.6359    0.6694    0.7137    0.7534    0.7791    0.8031    0.8164    0.8279    0.8361    0.8450    0.8478    0.8423    0.8405    0.8434   0.8446    0.8553    0.8795    0.9384    0.9844    1.0000    0.9959    0.9780    0.9616    0.9363    0.9086    0.8920    0.8577    0.8300    0.7900    0.7568    0.7263    0.6765    0.6486];

t = 1:nT;
plot(t,data(1:nT),'--*');

