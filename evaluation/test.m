% SBU
% mask_path = 'D:\Detection\compare_others\SBU\ShadowMasks\';
% pred_path = 'D:\Detection\compare_others\SBU\Zhu-SBU\';

% ISTD
% mask_path = 'D:\Detection\compare_others\ISTD\ShadowMasks\';
% pred_path = 'D:\Detection\compare_others\ISTD\Zhu-ISTD\';

% UCF
mask_path = 'D:\Detection\compare_others\UCF\ShadowMasks\';
pred_path = 'D:\Detection\compare_others\UCF\Zhu-UCF\';

[acc_final, final_BER, pErr, nErr, stats] = ComputeBERonSet(mask_path, pred_path);
%%%%% weighted F-measure
weighted_F_score = weighted_F_dataset(mask_path, pred_path);
fprintf('%s-- wFb: %.2f, BER: %.2f, pErr: %.2f, nErr: %.2f, acc: %.4f\n', pred_path, mean2(weighted_F_score)*100, final_BER, pErr, nErr, acc_final);
% fprintf('%s-- BER: %.2f, pErr: %.2f, nErr: %.2f, acc: %.4f\n', pred_path_num, final_BER, pErr, nErr, acc_final);
