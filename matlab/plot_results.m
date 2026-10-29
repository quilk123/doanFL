% Vẽ kết quả testbed (bước 8, so sánh reference_sim, đánh giá tấn công) — chạy được trên MATLAB và GNU Octave.
%   1. Xuất số liệu:  sh tools/export_matlab.sh   (-> experiments/matlab/*.csv)
%   2. MATLAB: mở thư mục testbed/matlab, chạy  plot_results
%      Octave (Docker, tự động): sh tools/run_plots.sh
% Hình: experiments/matlab/fig/*.png (+ .fig khi chạy MATLAB)
% Vẽ riêng một lần demo: plot_results('demo', 'demo/<thời điểm>/', '<thư mục hình>')  (tools/demo.sh tự gọi)
% Vẽ GỘP nhiều lần demo: plot_results('demo', {'demo/<lần 1>/', 'demo/<lần 2>/'}, '<thư mục hình>')
%   (vd. một lần có UAV + một lần không UAV -> mỗi chế độ kênh một ô, hình tốc độ có 2 đường)
%   source: 'step8' (mặc định) hoặc 'demo'; prefix: chuỗi hoặc danh sách chuỗi — chỉ lấy cấu hình có tên bắt đầu bằng
%   một trong các chuỗi này; FIG: nơi lưu hình
function plot_results(source, prefix, FIG)
  if nargin < 1, source = 'step8'; end
  if nargin < 2, prefix = ''; end
  here = fileparts(mfilename('fullpath'));
  D = fullfile(here, '..', 'experiments', 'matlab');
  if nargin < 3, FIG = fullfile(D, 'fig'); end
  if ~exist(FIG, 'dir'), mkdir(FIG); end
  isOctave = exist('OCTAVE_VERSION', 'builtin') ~= 0;
  POL = {'no_he', 'fixed10', 'fas10', 'fas50', 'adaptive', 'full_he'};   % thứ tự cố định trên mọi hình (fixed10 = phương án chính)
  COL = [0.45 0.45 0.45; 0.85 0.33 0.10; 0.73 0.46 0.09; 0.55 0.43 0.12; 0.11 0.62 0.46; 0.20 0.20 0.20];
  MRK = {'o', 's', 'd', 'p', '^', 'v'};

  R = read_rounds(fullfile(D, 'rounds.csv'));
  S = strcmp(R.source, source);
  if ischar(prefix), prefix = {prefix}; end
  prefix = prefix(~cellfun(@isempty, prefix));
  if ~isempty(prefix)
    keep = false(size(S));
    for k = 1:numel(prefix)   % Octave: strncmp không nhận N = 0 (đã bỏ chuỗi rỗng)
      keep = keep | strncmp(R.config, prefix{k}, numel(prefix{k}));
    end
    S = S & keep;
  end
  if ~any(S), error('không có vòng nào với source = %s, prefix = %s trong rounds.csv', source, strjoin(prefix, ' ')); end
  grp = unique(strcat(R.mode(S), '_', R.split(S)));
  % tỉ lệ khác (fixed20, fixed30, …) có trong số liệu: thêm vào cuối danh sách, mỗi tỉ lệ một màu riêng
  extra = setdiff(unique(R.policy(S)), POL);
  if ~isempty(extra)
    POL = [POL, reshape(extra, 1, [])];
    COL = [COL; [0.16 0.47 0.84; 0.38 0.31 0.84; 0.91 0.48 0.64; 0.0 0.51 0.0; 0.89 0.29 0.28](1:min(5, numel(extra)), :)];
    COL = [COL; repmat([0.3 0.3 0.3], max(0, numel(POL) - size(COL, 1)), 1)];
    MRK = [MRK, repmat({'o'}, 1, numel(extra))];
  end
  fprintf('[plot] %d vòng, nhóm: %s\n', sum(S), strjoin(grp', ', '));

  % 1-4: theo nhóm (chế độ kênh, phân hoạch), mỗi policy một đường
  specs = {
    'acc_vs_round', 't',            'acc',      100, 'Vòng',                       'Độ chính xác (%)'
    'acc_vs_time',  'cum_T_pi4_s',  'acc',      100, 'Thời gian tích lũy, thang Pi 4 (phút)', 'Độ chính xác (%)'
    'p_schedule',   't',            'p',        1,   'Vòng',                       'Tỷ lệ mã hóa HE p'
    'round_time',   't',            'T_pi4_s',  1,   'Vòng',                       'Thời gian vòng, thang Pi 4 (s)'
  };
  for s = 1:size(specs, 1)
    f = newfig(isOctave, 4.2 * numel(grp), 3.6);
    for g = 1:numel(grp)
      subplot(1, numel(grp), g); hold on; box on; grid on;
      h = []; lab = {}; ymax = 0;
      for k = 1:numel(POL)
        m = S & strcmp(strcat(R.mode, '_', R.split), grp{g}) & strcmp(R.policy, POL{k});
        if ~any(m), continue; end
        [~, o] = sort(R.t(m)); x = R.(specs{s, 2})(m); y = R.(specs{s, 3})(m) * specs{s, 4};
        x = x(o); y = y(o); ymax = max([ymax; y(:)]);
        if strcmp(specs{s, 2}, 'cum_T_pi4_s'), x = x / 60; end
        if strcmp(specs{s, 1}, 'p_schedule')
          h(end + 1) = stairs(x, y, 'Color', COL(k, :), 'LineWidth', 1.0 + 0.5 * (k == 2));    % fixed10 (chính) đậm hơn
        else
          h(end + 1) = plot(x, y, ['-' MRK{k}], 'Color', COL(k, :), 'MarkerSize', 4, 'LineWidth', 1.2);
        end
        if strcmp(specs{s, 1}, 'round_time')   % đánh dấu vòng trễ hạn
          L = logical(R.late(m)); L = L(o);
          plot(x(L), y(L), 'x', 'Color', [0.8 0 0], 'MarkerSize', 7, 'LineWidth', 1.4);
        end
        lab{end + 1} = strrep(POL{k}, '_', '\_');
      end
      if ~strcmp(specs{s, 2}, 'cum_T_pi4_s'), xlim([0.5, max(R.t(S)) + 0.5]); end
      if strcmp(specs{s, 1}, 'p_schedule'), ylim([-0.05 1.05]); end
      if strcmp(specs{s, 1}, 'round_time')
        tm = R.t_max_s(S); yl = ylim; xl = xlim;
        plot(xl, [tm(1) tm(1)], '--', 'Color', [0.8 0 0]);
        text(xl(2), tm(1), sprintf('t_{max} = %.1f s ', tm(1)), 'Color', [0.8 0 0], ...
             'VerticalAlignment', 'top', 'HorizontalAlignment', 'right');
        ylim([0 max([yl(2), tm(1) * 1.2, ymax * 1.45])]);   % chừa chỗ cho khung chú thích (góc trên trái)
      end
      title(strrep(grp{g}, '_', '\_')); xlabel(specs{s, 5}); ylabel(specs{s, 6});
      loc = 'northwest';                                          % Octave chưa có 'best'
      if strcmp(specs{s, 3}, 'acc'), loc = 'southeast'; end
      if g == 1 && ~isempty(h), legend(h, lab, 'Location', loc); end
    end
    save_fig(f, FIG, specs{s, 1}, isOctave);
  end

  % 5: kiểm chứng bóp tốc độ — tốc độ đo thật qua uav so với R lý thuyết (đường y = x)
  f = newfig(isOctave, 5, 4.4); hold on; box on; grid on;
  Rl = [R.R1_Mbps(S); R.R2_Mbps(S)]; Rm = [R.thr1_Mbps(S); R.thr2_Mbps(S)]; ok = ~isnan(Rl) & ~isnan(Rm) & Rl > 0;
  loglog(Rl(ok), Rm(ok), 'o', 'Color', [0.10 0.35 0.75], 'MarkerSize', 4);
  lim = [min(Rl(ok)) * 0.7, max(Rl(ok)) * 1.4];
  loglog(lim, lim, 'k--'); set(gca, 'XScale', 'log', 'YScale', 'log'); xlim(lim); ylim(lim);
  tk = [0.001 0.003 0.01 0.03 0.1 0.2 0.5 1 2 5 10]; tk = tk(tk >= lim(1) & tk <= lim(2));
  set(gca, 'XTick', tk, 'YTick', tk, 'XTickLabel', cellstr(num2str(tk')), 'YTickLabel', cellstr(num2str(tk')));
  dev = 100 * max(abs(Rm(ok) ./ Rl(ok) - 1));
  title(sprintf('Tốc độ đo qua uav vs lý thuyết (lệch max %.3f%%)', dev));
  xlabel('R lý thuyết (Mbit/s)'); ylabel('Tốc độ đo (Mbit/s)');
  save_fig(f, FIG, 'rate_validation', isOctave);

  % 6: tốc độ kênh CÓ UAV (uav_noma_pc) và KHÔNG UAV (direct_noma) — hàm phân phối thực nghiệm
  f = newfig(isOctave, 5.4, 4); hold on; box on; grid on;
  modes = unique(R.mode(S)); h = []; lab = {};
  cm = [0.10 0.35 0.75; 0.85 0.33 0.10; 0.3 0.6 0.3; 0.5 0.5 0.5];
  for k = 1:numel(modes)
    m = S & strcmp(R.mode, modes{k});
    v = sort([R.R1_Mbps(m); R.R2_Mbps(m)]); v = v(v > 0);
    h(end + 1) = stairs(v, (1:numel(v))' / numel(v), 'Color', cm(k, :), 'LineWidth', 1.6);
    lab{end + 1} = sprintf('%s (trung vị %.3g Mbit/s)', strrep(modes{k}, '_', '\_'), median(v));
  end
  set(gca, 'XScale', 'log'); xlabel('Tốc độ kênh R (Mbit/s)'); ylabel('Tỷ lệ vòng × thiết bị ≤ R');
  title('Tốc độ kênh: có UAV và không UAV'); legend(h, lab, 'Location', 'northwest');
  save_fig(f, FIG, 'rate_uav_vs_direct', isOctave);

  % 7: tỷ lệ vòng trễ hạn theo policy, mỗi nhóm một cột
  f = newfig(isOctave, 6, 4); late = nan(numel(POL), numel(grp));
  for g = 1:numel(grp)
    for k = 1:numel(POL)
      m = S & strcmp(strcat(R.mode, '_', R.split), grp{g}) & strcmp(R.policy, POL{k});
      if any(m), late(k, g) = 100 * mean(R.late(m)); end
    end
  end
  bar(late); box on; grid on; set(gca, 'XTick', 1:numel(POL), 'XTickLabel', strrep(POL, '_', '\_'));
  ylabel('Vòng trễ hạn (%)'); title('Tỷ lệ trễ hạn so với t_{max}');
  legend(strrep(grp, '_', '\_'), 'Location', 'northwest');
  save_fig(f, FIG, 'late_pct', isOctave);

  % 8: testbed vs reference_sim (cùng lịch p)
  fn = fullfile(D, 'ref_compare.csv');
  if strcmp(source, 'step8') && exist(fn, 'file')   % hình 8–9 chỉ cho số liệu bước 8
    C = read_csv(fn, '%s%f%f%f%f%f', {'config', 't', 'p', 'acc_testbed', 'acc_ref', 'diff_pt'});
    cf = unique(C.config); f = newfig(isOctave, 3.2 * numel(cf), 3.4);
    for k = 1:numel(cf)
      subplot(1, numel(cf), k); hold on; box on; grid on;
      m = strcmp(C.config, cf{k});
      plot(C.t(m), 100 * C.acc_testbed(m), 'o-', 'MarkerSize', 3, 'Color', [0.10 0.35 0.75]);
      plot(C.t(m), 100 * C.acc_ref(m), 's--', 'MarkerSize', 3, 'Color', [0.85 0.33 0.10]);
      title(strrep(strrep(cf{k}, 'uav_noma_pc_', ''), '_', '\_')); xlabel('Vòng');
      if k == 1, ylabel('Độ chính xác (%)'); legend({'testbed', 'reference\_sim'}, 'Location', 'southeast'); end
    end
    save_fig(f, FIG, 'ref_compare', isOctave);
  elseif strcmp(source, 'step8')
    fprintf('[plot] chưa có ref_compare.csv — bỏ qua hình so sánh reference_sim\n');
  end

  % 9: đánh giá tấn công — độ khôi phục cập nhật của bs (vòng 1)
  fn = fullfile(D, 'attack.csv');
  if strcmp(source, 'step8') && exist(fn, 'file')
    % device là chữ: '1', '2' hoặc '1&2' (dòng tương quan cặp do π chung)
    A = read_csv(fn, '%s%s%s%s%f%f%f%f%f%f%f', {'config', 'policy', 'split', 'device', 'p', 'recovery_corr', ...
                 'auc_bs', 'auc_null_halfwidth95', 'deshuffle_exact', 'deshuffle_corr', 'pair_corr_same_pi'});
    sp = unique(A.split); rc = nan(numel(POL), numel(sp)); au = rc;
    for g = 1:numel(sp)
      for k = 1:numel(POL)
        m = strcmp(A.split, sp{g}) & strcmp(A.policy, POL{k}) & ~strcmp(A.device, '1&2');   % bỏ dòng tương quan cặp
        if any(m), rc(k, g) = 100 * mean(A.recovery_corr(m)); au(k, g) = mean(A.auc_bs(m)); end
      end
    end
    f = newfig(isOctave, 9, 3.8);
    subplot(1, 2, 1); bar(rc); box on; grid on; set(gca, 'XTick', 1:numel(POL), 'XTickLabel', strrep(POL, '_', '\_'));
    ylabel('Độ khôi phục cập nhật (%)'); title('bs khôi phục Δ (vòng 1)'); legend(sp, 'Location', 'northeast');
    subplot(1, 2, 2); bar(au); hold on; box on; grid on; set(gca, 'XTick', 1:numel(POL), 'XTickLabel', strrep(POL, '_', '\_'));
    hw = A.auc_null_halfwidth95(~isnan(A.auc_null_halfwidth95));
    if ~isempty(hw), plot(xlim, 0.5 + hw(1) * [1 1], 'r--'); end
    ylim([0.45 0.6]); ylabel('AUC suy luận thành viên'); title('AUC (đường đỏ: ngưỡng ý nghĩa 95%)');
    save_fig(f, FIG, 'attack', isOctave);
  elseif strcmp(source, 'step8')
    fprintf('[plot] chưa có attack.csv — bỏ qua hình tấn công\n');
  end
  fprintf('[plot] xong -> %s\n', FIG);
end


function R = read_rounds(fn)
  names = {'source', 'config', 'mode', 'split', 'policy', 'time_scale', 't', 'p', 'n_ct', 'n_he', ...
           'R1_Mbps', 'R2_Mbps', 'thr1_Mbps', 'thr2_Mbps', 'upload_MB_per_dev', 'tx1_s', 'tx2_s', ...
           'T_pred_s', 'T_pi4_s', 'cum_T_pi4_s', 't_max_s', 't_select_s', 'late', 'late_pred', ...
           'acc', 'loss', 'meas_ok', 'train_slowdown_pct'};
  R = read_csv(fn, [repmat('%s', 1, 5) repmat('%f', 1, numel(names) - 5)], names);
end


function T = read_csv(fn, fmt, names)
  % textscan: chạy giống nhau trên MATLAB và Octave (Octave không có readtable). Kiểm tra tiêu đề khớp tên cột.
  fid = fopen(fn, 'r', 'n', 'UTF-8');
  hdr = strtrim(fgetl(fid));
  if ~strcmp(hdr, strjoin(names, ','))
    fclose(fid); error('tiêu đề %s khác mong đợi: %s', fn, hdr);
  end
  c = textscan(fid, fmt, 'Delimiter', ',', 'EmptyValue', NaN);
  fclose(fid);
  for k = 1:numel(names), T.(names{k}) = c{k}; end
end


function f = newfig(isOctave, w, h)
  f = figure('Visible', 'off', 'Units', 'inches', 'Position', [1 1 w h], 'PaperPositionMode', 'auto');
  if isOctave, set(f, 'PaperUnits', 'inches', 'PaperPosition', [0 0 w h]); end
end


function save_fig(f, d, name, isOctave)
  print(f, fullfile(d, [name '.png']), '-dpng', '-r150');
  if ~isOctave, savefig(f, fullfile(d, [name '.fig'])); end
  close(f);
  fprintf('[plot] %s.png\n', name);
end
