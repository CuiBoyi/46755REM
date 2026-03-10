function T = buildLoadBids(totalMWh)

% ---- Input: total system energy over the horizon ----
% Example: totalMWh = 2500;

% Table 4 data (Load#, Node, share%)
load_id = (1:17)';
node    = [1 2 3 4 5 6 7 8 9 10 13 14 15 16 18 19 20]';
share   = [3.8 3.4 6.3 2.6 2.5 4.8 4.4 6.0 6.1 6.8 9.3 6.8 11.1 3.5 11.7 6.4 4.5]';

% ---- Convert share -> MWh ----
E = totalMWh .* share ./ 100;

% ---- Assume demand bid prices ($/MWh) ----
pmin = 15;  % example
pmax = 20*1.3;  % example

smin = min(share);
smax = max(share);

price = pmax - (pmax - pmin) .* (share - smin) ./ (smax - smin);

% ---- Output table ----
T = table(load_id, node, share, E, price, ...
    'VariableNames', {'Load','Node','Share_percent','Energy_MWh','BidPrice_EUR_per_MWh'});

disp(T)

% Optional plot
figure;
plot(load_id, price, '--o', 'LineWidth', 1.5);
grid on; box on;
xlabel('Load #');
ylabel('Assumed bid price (€/MWh)');
title('Assumed demand bid prices by load');
end

buildLoadBids(2650.5)


function plotDemandCurve(totalMWh)

% ---------------- Load Data ----------------
load_id = (1:17)';
node    = [1 2 3 4 5 6 7 8 9 10 13 14 15 16 18 19 20]';
share   = [3.8 3.4 6.3 2.6 2.5 4.8 4.4 6.0 6.1 6.8 9.3 6.8 11.1 3.5 11.7 6.4 4.5]';

% ---------------- Energy ----------------
E = totalMWh .* share ./ 100;

% ---------------- Assumed Bid Prices ----------------
pmin = 15;   % €/MWh
pmax = 20*1.3;   % €/MWh

smin = min(share);
smax = max(share);

price = pmax - (pmax - pmin) .* (share - smin) ./ (smax - smin);

% ---------------- Sort by price (descending) ----------------
[price_sorted, idx] = sort(price, 'descend');
E_sorted = E(idx);

% ---------------- Cumulative Demand ----------------
cumDemand = cumsum(E_sorted);

% ---------------- Plot Demand Curve ----------------
figure;
stairs([0; cumDemand], [price_sorted; price_sorted(end)], ...
       'LineWidth', 2);

grid on; box on;
xlabel('Cumulative Demand (MWh)');
ylabel('Bid Price (€/MWh)');
ylim([0, pmax]);
title('Market Demand Curve');
set(gca,'XLim',[0 totalMWh]);

end

plotDemandCurve(2650.5)