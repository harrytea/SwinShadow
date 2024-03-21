% SBU
% mask_path = 'D:\ShadowData\SBU\SBU-Test\ShadowMasks\';
% pred_path = 'E:\VideoDetection\SID\seg15_bootce\sbu\';

% ISTD
mask_path = 'D:\ShadowData\ISTD\test\test_B\';
pred_path = 'E:\VideoDetection\SID\seg15_bootce\istd\';

% UCF
% mask_path = 'D:\ShadowData\UCF\GroundTruth\';
% pred_path = 'E:\VideoDetection\SID\results\ucf\';

for iter=0:1:40  % start:step:end
    pred_path_num = sprintf("%s%d%s", pred_path, iter, '\');  % corresponding scan image path
    % pred_path_num = fullfile(pred_path, iter) 
    pred_path_num = char(pred_path_num);
    [acc_final, final_BER, pErr, nErr, stats] = ComputeBERonSet(mask_path, pred_path_num);
    %%%%% weighted F-measure
    % weighted_F_score = weighted_F_dataset(mask_path, pred_path_num);
    % fprintf('%s-- wFb: %.2f, BER: %.2f, pErr: %.2f, nErr: %.2f, acc: %.4f\n', pred_path_num, mean2(weighted_F_score)*100, final_BER, pErr, nErr, acc_final);
    fprintf('%s-- BER: %.2f, pErr: %.2f, nErr: %.2f, acc: %.4f\n', pred_path_num, final_BER, pErr, nErr, acc_final);
end