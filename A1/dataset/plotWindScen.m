%% Matlab script to plot wind scenarios
% This script needs WindScen.mat file in the same directory
% zone: zones can be a number or array, due to data limitations the values should be in 1<=zone<=15
% nS  : number of scenarios to plot
% nT  : number of time periods to plot (this is range of x-axis), due to
%       data limitations this number should be less than 43
% Examples
% (1) To plot two scenarios for zone 1 and for time period [1,43] we do the
% following:
% plotWindScen(1,2,43)
% (2) To plot twenty scenarios for zone 1 and 4 for time period [1,43] we
% do the following:
% plotWindScen([1,4],20,43)
% W. Bukhsh, May 2014
% wbukhsh@gmail.com

function plotWindScen(zone,nS,nT)

load WindScen.mat

for p = 1:numel(zone)
    figure(p)
    clf
    t = 1:nT;
    clear Scen
    Scen = WindScen{zone(p)}(1:nT,1:nS);
    plot(t,Scen,'--*')
end
