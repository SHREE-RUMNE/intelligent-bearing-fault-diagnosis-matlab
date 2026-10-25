%% ========================================================================
%  STEP 4.7: LIVE REAL-TIME STREAMING EDGE MONITOR ON KAGGLE CWRU FILES
%  ========================================================================
clearvars -except dataTable mdlA sevMdl brg bandLimits fs;
if ~exist('mdlA', 'var')
    load('Project5_Trained_Workspace.mat');
end

classNames = {'Healthy', 'Inner-Race', 'Outer-Race', 'Ball', 'Gear-Tooth'};
classes = categorical(classNames, classNames);

idxH  = find(dataTable.FaultType == 'Healthy'    & dataTable.Split == "Test", 4, 'first');
idxIR = find(dataTable.FaultType == 'Inner-Race' & dataTable.FaultSize == 0.18 & dataTable.Split == "Test", 4, 'first');
idxB  = find(dataTable.FaultType == 'Ball'       & dataTable.FaultSize == 0.36 & dataTable.Split == "Test", 4, 'first');
idxOR = find(dataTable.FaultType == 'Outer-Race' & dataTable.FaultSize == 0.53 & dataTable.Split == "Test", 4, 'first');

continuousSignal = vertcat(dataTable.Signal{[idxH; idxIR; idxB; idxOR]});
rpmStream = 1772;
fr = rpmStream / 60;

buf  = dsp.AsyncBuffer('Capacity', 4 * 12000);
vote = [1, 0, 0, 0, 0];

fig = figure('Name', 'Live Edge AI Drivetrain Monitor', 'Color', [0.09 0.11 0.15], 'Position', [70 70 1260 680]);
tiledlayout(2, 2, 'TileSpacing', 'compact', 'Padding', 'compact');

ax1 = nexttile;
hWave = plot(ax1, (0:fs-1)/fs, zeros(fs,1), 'Color', [0 0.85 1], 'LineWidth', 0.8);
set(ax1, 'Color', [0.14 0.16 0.22], 'XColor', 'w', 'YColor', 'w'); ylim(ax1, [-3.5 3.5]); grid on;
title(ax1, '1. Live CWRU Accelerometer Window (1s Window, 50% Overlap)', 'Color', 'w');
xlabel(ax1, 'Window Time (s)'); ylabel(ax1, 'Acceleration (g)');

ax2 = nexttile;
hSpec = plot(ax2, 0:7, zeros(1,8), 'Color', [1 0.78 0.1], 'LineWidth', 1.4); hold on;
xline(brg.Order_BPFO, '--r', 'BPFO'); xline(brg.Order_BPFI, '--g', 'BPFI'); xline(brg.Order_BSF, '--m', 'BSF');
set(ax2, 'Color', [0.14 0.16 0.22], 'XColor', 'w', 'YColor', 'w'); xlim([0 7.5]); ylim([0 0.45]); grid on;
title(ax2, '2. Live Order-Tracked Hilbert Envelope Spectrum', 'Color', 'w');
xlabel(ax2, 'Shaft Order (Multiples of Shaft Speed f_r)'); ylabel(ax2, 'Envelope Energy');

ax3 = nexttile;
hBar = bar(classes, vote * 100, 'FaceColor', [0.2 0.75 0.4]);
set(ax3, 'Color', [0.14 0.16 0.22], 'XColor', 'w', 'YColor', 'w'); ylim([0 105]); grid on;
yline(70, '--r', 'Alert Threshold (70%)', 'LineWidth', 1.2);
title(ax3, '3. Exponentially Smoothed Diagnosis Confidence (%)', 'Color', 'w');
ylabel(ax3, 'Confidence (%)');

ax4 = nexttile; axis off;
hStatus = text(0.5, 0.58, 'STREAM STARTING...', 'Color', 'w', 'FontSize', 14, 'FontWeight', 'bold', ...
    'HorizontalAlignment', 'center', 'BackgroundColor', [0.18 0.55 0.25], 'EdgeColor', 'w', 'LineWidth', 1.5, 'Margin', 14);
hMetrics = text(0.5, 0.18, '', 'Color', [0.85 0.9 1], 'FontSize', 11, 'HorizontalAlignment', 'center', 'FontName', 'FixedWidth');

chunkSize = 1200;
numChunks = floor(length(continuousSignal) / chunkSize);
elapsedTime = 0;
[b_bp, a_bp] = butter(4, bandLimits / (fs/2), 'bandpass');

for k = 1:numChunks
    if ~isvalid(fig), break; end
    write(buf, continuousSignal((k-1)*chunkSize + 1 : k*chunkSize));
    elapsedTime = elapsedTime + 0.1;

    if buf.NumUnreadSamples >= 12000
        tStart = tic;
        x = read(buf, 12000, 6000);
        x_bp = filtfilt(b_bp, a_bp, x);
        env  = abs(hilbert(x_bp));
        env0 = env - mean(env);

        N = length(env0);
        Y = abs(fft(env0)) / N;
        Y_half = 2 * Y(1:floor(N/2)+1);
        order_axis = ((0:floor(N/2))' * (fs / N)) / fr;

        getOrdE = @(ord) max(Y_half(abs(order_axis - ord) <= 0.14)) + ...
                         0.5 * max(Y_half(abs(order_axis - 2*ord) <= 0.14));
        featVec = [getOrdE(brg.Order_BPFO), getOrdE(brg.Order_BPFI), ...
                   getOrdE(brg.Order_BSF),  getOrdE(brg.Order_Gear), ...
                   rms(x), kurtosis(x), peak2rms(x), skewness(x)];
        
        cPred = predict(mdlA, featVec);
        inferMs = toc(tStart) * 1000;

        onehot = double(classes == cPred);
        vote = 0.75 * vote + 0.25 * onehot;
        [conf, lblIdx] = max(vote);
        activeClass = classes(lblIdx);

        if activeClass == 'Healthy'
            sevStr = 'NONE (0.00 mm - Normal Operation)';
        else
            estMm = predict(sevMdl, array2table(featVec, 'VariableNames', ...
                    {'eBPFO','eBPFI','eBSF','eGear','RMS','Kurtosis','CrestFactor','Skewness'}));
            sevCat = discretize(estMm, [0, 0.25, 0.45, Inf], 'categorical', {'INCIPIENT (0.18mm)', 'MODERATE (0.36mm)', 'SEVERE (0.53mm)'});
            sevStr = sprintf('%s [Est: %.2f mm]', string(sevCat), estMm);
        end

        set(hWave, 'YData', x);
        set(hSpec, 'XData', order_axis(order_axis <= 7.5), 'YData', Y_half(order_axis <= 7.5));
        set(hBar, 'YData', vote * 100);

        if activeClass ~= 'Healthy' && conf > 0.70
            set(hBar, 'FaceColor', [0.9 0.25 0.2]);
            set(hStatus, 'String', sprintf('⚠ FAULT ALERT: %s\nSeverity: %s', upper(string(activeClass)), sevStr), ...
                         'BackgroundColor', [0.75 0.15 0.15]);
        else
            set(hBar, 'FaceColor', [0.2 0.75 0.4]);
            set(hStatus, 'String', sprintf('✓ DRIVETRAIN HEALTHY\nSource: Time_Normal_1_098.mat'), ...
                         'BackgroundColor', [0.18 0.55 0.25]);
        end
        set(hMetrics, 'String', sprintf('Stream Time: %4.1f s | Speed: %d RPM | Confidence: %5.1f%% | Edge Latency: %.2f ms', ...
            elapsedTime, rpmStream, conf*100, inferMs));
        drawnow;
        pause(0.18);
    end
end