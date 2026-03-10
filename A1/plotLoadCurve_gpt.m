%% Plot Load Curve Function
% This function plots normalized demand data.
% The maximum available time horizon is 48 periods.
% Example usage:
%   plotLoadCurve(24)
%
% Author: W. Bukhsh (Original)
% Improved version with formatting and labels

function plotLoadCurve(nT)

    %% ---------------- Input Validation ----------------
    if nargin == 0
        error('You must specify the number of time periods (nT).');
    end

    if nT > 48
        error('Due to data limitations, nT must not exceed 48.');
    end

    if nT <= 0
        error('nT must be a positive integer.');
    end

    %% ---------------- Demand Data ----------------
    data = [ ...
        0.6642 0.6645 0.6517 0.6293 0.6144 0.6074 ...
        0.5935 0.5791 0.5637 0.5566 0.5553 0.5612 ...
        0.5850 0.5993 0.6225 0.6359 0.6694 0.7137 ...
        0.7534 0.7791 0.8031 0.8164 0.8279 0.8361 ...
        0.8450 0.8478 0.8423 0.8405 0.8434 0.8446 ...
        0.8553 0.8795 0.9384 0.9844 1.0000 0.9959 ...
        0.9780 0.9616 0.9363 0.9086 0.8920 0.8577 ...
        0.8300 0.7900 0.7568 0.7263 0.6765 0.6486 ];

    %% ---------------- Time Vector ----------------
    t = 1:nT;

    %% ---------------- Plot ----------------
    figure;
    plot(t, data(1:nT), '--o', ...
        'LineWidth', 1.8, ...
        'MarkerSize', 6);

    grid on;
    box on;

    xlabel('Time Period', 'FontSize', 12);
    ylabel('Normalized Demand', 'FontSize', 12);
    title('Daily Load Curve', 'FontSize', 14);

    xlim([1 nT]);
    ylim([min(data)-0.02 max(data)+0.02]);

end

plotLoadCurve(24)