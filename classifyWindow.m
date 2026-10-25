function label = classifyWindow(x) %#codegen
% CLASSIFYWINDOW Standalone entry-point for MATLAB Coder (Step 4.6)
    rmsVal   = sqrt(mean(x.^2));
    kurtVal  = kurtosis(x);
    crestVal = max(abs(x)) / max(eps, rmsVal);
    
    N = length(x);
    X_mag = abs(fft(x));
    X_half = X_mag(1:floor(N/2));
    peakEnergy = max(X_half) / (sum(X_half) + eps);

    if rmsVal < 0.25 && kurtVal < 3.2
        label = categorical({'Healthy'});
    elseif kurtVal > 5.5 && crestVal > 4.5
        label = categorical({'Inner-Race'});
    elseif peakEnergy > 0.15
        label = categorical({'Outer-Race'});
    elseif rmsVal > 0.8
        label = categorical({'Ball'});
    else
        label = categorical({'Gear-Tooth'});
    end
end