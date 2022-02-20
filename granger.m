clear
path_cluster = '/autofs/cluster/transcend/jussi/';
addpath(sprintf('%s/tools/fieldtrip-20210929/', path_cluster))

data_ID = 'ACseedConClusters';
conds = {'MSS_SWS'};

tmin = -0.5;
tmax = 1.5;
tw = 2.0;
fmin = 0;
fmax = 100;
fstep = 1;
toi = tmin:tw:tmax;
foi = fmin:fstep:fmax;
taper = 'dpss';
tapsmofrq = 2;
n_tws = length(toi)-1;
add_bl_contrast = 0;
noise_bl = 0; % baselines followed by noise stimuli only

% read subject and group IDs
f = sprintf('%s/sub_IDs.txt', path_cluster);
fid = fopen(f, 'r');
sub_IDs = {};
while ~feof(fid)
    sub_IDs{end+1,1} = fgetl(fid);
end
fclose(fid);

f = sprintf('%s/group_IDs.txt', path_cluster);
fid = fopen(f, 'r');
group_IDs = {};
while ~feof(fid)
    group_IDs{end+1,1} = fgetl(fid);
end
fclose(fid);
% inds_TD = find(strcmp(group_IDs, 'TD'));
% inds_ASD = find(strcmp(group_IDs, 'ASD'));

n_subs = length(sub_IDs);
n_conds = length(conds);
n_freqs = length(foi);

% con_data = zeros(n_subs, n_conds, n_cons, n_freqs);

%% calculate granger
for i_sub=1:n_subs
    for i_cond=1:n_conds
        % read epochs data
        f = sprintf('%s/%s/%s_%s_-500-2000ms_0-200Hz-epo.fif', ...
                    path_cluster, sub_IDs{i_sub}, data_ID, conds{i_cond});
        cfg = [];
        cfg.dataset = f;
        data = ft_preprocessing(cfg);
        
        for i_toi=1:n_tws
        
            % crop to time window of interest
            cfg = [];
            cfg.toilim = [toi(i_toi) toi(i_toi+1)];
            this_data = ft_redefinetrial(cfg, data);

            % frequency analysis
            cfg = [];
            cfg.method = 'mtmfft';
            cfg.output = 'fourier';
            cfg.taper = taper;
            cfg.tapsmofrq = tapsmofrq;
            cfg.keeptrials = 'yes';
            cfg.foi = foi;
            cfg.pad = round(tw)+0.001;
            cfg.padtype = 'zero';
            freq = ft_freqanalysis(cfg, this_data);

            % granger causality analysis
            cfg = [];
            cfg.method = 'granger';
            for i=1:length(this_data.label)-1
                cfg.channelcmb{i,1} = this_data.label{1};
                cfg.channelcmb{i,2} = this_data.label{i+1};
            end
            cfg.granger.sfmethod = 'bivariate';
            con = ft_connectivityanalysis(cfg, freq);

            % put data into array
            con_data(i_sub, i_cond, i_toi, :, :) = con.grangerspctrm;
        end
        
        if noise_bl && strcmp(conds{i_cond}, 'Noise')
            
            % crop to time window of interest
            cfg = [];
            cfg.toilim = [data.time{1}(end)-0.5 data.time{1}(end)];
            this_data = ft_redefinetrial(cfg, data);

            % frequency analysis
            cfg = [];
            cfg.method = 'mtmfft';
            cfg.output = 'fourier';
            cfg.taper = taper;
            cfg.tapsmofrq = tapsmofrq;
            cfg.keeptrials = 'yes';
            cfg.foi = foi;
            cfg.pad = round(tw);
            cfg.padtype = 'zero';
            freq = ft_freqanalysis(cfg, this_data);

            % granger causality analysis
            cfg = [];
            cfg.method = 'granger';
            for i=1:length(this_data.label)-1
                cfg.channelcmb{i,1} = this_data.label{1};
                cfg.channelcmb{i,2} = this_data.label{i+1};
            end
            cfg.granger.sfmethod = 'bivariate';
            con = ft_connectivityanalysis(cfg, freq);
            con_bl(i_sub, 1, 1, :, :) = con.grangerspctrm;
        end        
    end
end

con_labels = {};
for i=1:length(con.labelcmb)
    roi1 = split(con.labelcmb{i,1}, '[');
    roi1 = roi1{1};
    roi2 = split(con.labelcmb{i,2}, '[');
    roi2 = roi2{1};
    con_labels{i} = sprintf('%s-%s', roi1, roi2);

end
n_cons = length(con_labels);

%% Do contrasts
contrasts = {};
contrast_labels = {};
for i=1:n_conds-1
    for j=i+1:n_conds
        contrasts{end+1} = con_data(:,i,:,:,:)-con_data(:,j,:,:,:);
        contrast_labels{end+1} = sprintf('%s-%s', conds{i}, conds{j});
    end
end

% Add baseline contrasts
if add_bl_contrast
    if noise_bl
        bl = con_bl;
    else    
        bl = mean(con_data(:,:,1,:,:),2);
    end
    for i=1:n_conds
        contrasts{end+1} = con_data(:,i,:,:,:)-bl;
        contrast_labels{end+1} = sprintf('%s-BL', conds{i});
    end
end


% Add contrasts data
con_data = cat(2, con_data, cell2mat(contrasts));
conds = [conds, contrast_labels];
n_conds = length(conds);

%% Plot as a function of frequency

cois = {'MSS_SWS'};

coni = 'con_phte-8-AC';
i_con = find(strcmp(con_labels, coni));

fmin = 6;
fmax = 62;

c_TD = [0.56 0.93 0.56];
c_ASD = [0.86 0.44 0.84];
c2_TD = [0.243 0.588 0.318];
c2_ASD = [0.494 0.184 0.556];

inds_TD = find(strcmp(group_IDs, 'TD'));
inds_ASD = find(strcmp(group_IDs, 'ASD'));

% for i_con=1:n_cons

for i_tw=1:n_tws
    if length(cois)>1
        figure
        for i=1:length(cois)
            cond = cois{i};
            i_cond = find(strcmp(conds, cond));
            
            TD = squeeze(con_data(inds_TD, i_cond, i_tw, i_con, ...
                find(foi==fmin):find(foi==fmax)));
            ASD = squeeze(con_data(inds_ASD, i_cond, i_tw, i_con, ...
                find(foi==fmin):find(foi==fmax)));
            
            if i==1
                c_TD = [0.56 0.93 0.56];
                c_ASD = [0.86 0.44 0.84];
            else
                c_TD = [0.243 0.588 0.318];
                c_ASD = [0.494 0.184 0.556];
            end
            plot(fmin:fstep:fmax, mean(TD,1), 'LineWidth', 2, 'Color', c_TD)            
            hold on
            plot(fmin:fstep:fmax, mean(ASD,1), 'LineWidth', 2, 'Color', c_ASD)
            axis tight
            xticks(fmin+2:4:fmax-2)            
        end
%         title(sprintf('%s %s %.1f-%.1fs %s', cois{1}, cois{2}, toi(i_tw), toi(i_tw+1)));
        xlabel('Frequency (Hz)')
        ylabel('Granger causality')
        legend('TD MSS', 'ASD MSS', 'TD SWS', 'ASD SWS', 'Location', 'northeast')
        exportgraphics(gcf, ...
            sprintf('%s/figures/connectivity/granger_%s_%s_%.1f-%.1fs_%d-%dHz_%s.png', ...
                path_cluster, cois{1}, cois{2}, toi(i_tw), toi(i_tw+1), fmin, fmax, ...
                con_labels{i_con}), 'Resolution', 300)
        close

    else
        i_cond = find(strcmp(conds, cois{1}));
        cond = conds{i_cond};
        TD = squeeze(con_data(inds_TD, i_cond, i_tw, i_con, ...
            find(foi==fmin):find(foi==fmax)));
        ASD = squeeze(con_data(inds_ASD, i_cond, i_tw, i_con, ...
            find(foi==fmin):find(foi==fmax)));
        figure
        plot(fmin:fstep:fmax, mean(TD,1), 'LineWidth', 2, 'Color', c_TD)            
        hold on
        plot(fmin:fstep:fmax, mean(ASD,1), 'LineWidth', 2, 'Color', c_ASD)
        axis tight
        xticks(fmin+2:4:fmax-2)
        title(sprintf('%s %.1f-%.1fs %s', cond, toi(i_tw), toi(i_tw+1)));
        xlabel('Frequency (Hz)')
        ylabel('Granger causality')
        legend('TD', 'ASD', 'Location', 'northeast')
        exportgraphics(gcf, ...
            sprintf('%s/figures/connectivity/granger_%s_%.1f-%.1fs_%d-%dHz_%s.png', ...
                path_cluster, cond, toi(i_tw), toi(i_tw+1), fmin, fmax, ...
                con_labels{i_con}), 'Resolution', 300)
        close

    end
end
    

%% Average data into frequency bins

freq_bin_labels = {'8-12Hz', '8-30Hz', '12-30Hz', '18-24Hz', '18-30Hz', '8-60Hz', '60-100Hz'};
freq_bin_values = [[8 12]; [8 30]; [12 30]; [18 24]; [18 30]; [8 60]; [60 100]];
freq_bins = containers.Map;
for i=1:length(freq_bin_labels)
    freq_bins(freq_bin_labels{i}) = [freq_bin_values(i,1) freq_bin_values(i,2)]; 
end

n_freq_bins = length(freq_bins);
% con_data_binned = zeros(n_subs, n_conds, n_cons, n_freq_bins);
otherdims = repmat({':'},1,ndims(con_data)-1);
for i=1:n_freq_bins
    flims = freq_bins(freq_bin_labels{i});
    i_fmin = find(foi==flims(1));
    i_fmax = find(foi==flims(2));
    con_data_binned(otherdims{:},i) = mean(con_data(otherdims{:},i_fmin:i_fmax), ndims(con_data));
end


%% test for differences between groups

cois = {'MSS_SWS'};
% conis = {'AC-pCS4'};
% freq_bin = 'alpha1';
nonparametric = 1;

exclude = {}; % '052902', '090902', '048102', '075401', '096603'

this_data = con_data_binned;
this_group_IDs = group_IDs;
if ~isempty(exclude)
    inds_exclude = sort(cellfun(@(x) find(strcmp(sub_IDs, x)), exclude));
    this_data(inds_exclude,:,:,:) = [];
    this_group_IDs(inds_exclude) = [];
end

inds_TD = find(strcmp(this_group_IDs, 'TD'));
inds_ASD = find(strcmp(this_group_IDs, 'ASD'));
% i_freq_bin = find(strcmp(freq_bin_labels, freq_bin));

for i_coi=1:length(cois)
    i_cond = find(strcmp(conds, cois{i_coi}));
    for i_tw=1:n_tws
        for i_con=1:length(con_labels)
%             i_con = find(strcmp(con_labels, coni));
            for i_freq_bin=1:n_freq_bins
                this_freq_bin_label = freq_bin_labels{i_freq_bin};
                TD = this_data(inds_TD, i_cond, i_tw, i_con, i_freq_bin);
                ASD = this_data(inds_ASD, i_cond, i_tw, i_con, i_freq_bin);
    %             fprintf('\nTD mean: %.3f', mean(TD))
    %             fprintf('\nASD mean: %.3f', mean(ASD))
                if nonparametric
                    [p,~,stats] = ranksum(TD, ASD);
                    if p<0.2
                        fprintf('\n%s %s %s-%ss %s: z=%.2f, p=%.5f\n', conds{i_cond}, ... 
                                con_labels{i_con}, num2str(toi(i_tw)), ...
                                num2str(toi(i_tw+1)), this_freq_bin_label, ...
                                stats.zval, p);
                    end
                else
                    [~,p,~,stats] = ttest2(TD, ASD);
                    if p<0.1
                        fprintf('\n%s %s %s-%s %s: t=%.2f, p=%.5f\n', conds{i_cond}, ... 
                                con_labels{i_con}, num2str(toi(i_tw)), ...
                                num2str(toi(i_tw+1)), this_freq_bin_label, ...
                                stats.tstat, p);
                    end            
                end
            end
        end
    end
end

%% bar plots

cois = {'MSS_SWS'};
con = 'con_phte-8-AC';
freq_bin = '8-12Hz';
i_tw = 1;

exclude = {}; % '052902', '090902', '048102', '075401', '096603'

c_TD = [0.56 0.93 0.56];
c_ASD = [0.86 0.44 0.84];

this_data = con_data_binned;
this_group_IDs = group_IDs;
if ~isempty(exclude)
    inds_exclude = sort(cellfun(@(x) find(strcmp(sub_IDs, x)), exclude));
    this_data(inds_exclude,:,:,:) = [];
    this_group_IDs(inds_exclude) = [];
end

inds_TD = find(strcmp(this_group_IDs, 'TD'));
inds_ASD = find(strcmp(this_group_IDs, 'ASD'));

i_con = find(strcmp(con_labels, con));
i_freq_bin = find(strcmp(freq_bin_labels, freq_bin));

data_mean = nan(length(cois), 2);
data_sem = nan(length(cois), 2);
for i_coi=1:length(cois)
    i_cond = find(strcmp(conds, cois{i_coi}));
    TD = this_data(inds_TD, i_cond, i_tw, i_con, i_freq_bin);
    ASD = this_data(inds_ASD, i_cond, i_tw, i_con, i_freq_bin);
    data_mean(i_coi, 1) = mean(TD);
    data_mean(i_coi, 2) = mean(ASD);
    data_sem(i_coi, 1) = std(TD)/sqrt(length(TD));
    data_sem(i_coi, 2) = std(ASD)/sqrt(length(ASD));
end

% bar plot
b = bar(data_mean);
% b(1).FaceColor = c_TD;
% b(2).FaceColor = c_ASD;
% b(1).BarWidth = 1;
% b(2).BarWidth = 1;
title(sprintf('%s %s', con, freq_bin))
xlabel('Condition')
ylabel('Granger causality')
xticklabels(cois)
% legend('TD', 'ASD', 'Location', 'southeast')

% plot SEM error lines on the mean bars
hold on
for i=1:2
    errorbar(b(i).XEndPoints, data_mean(:,i), data_sem(:,i), '.', 'Color', 'k');
end

% save figure
f = gcf;
exportgraphics(f, sprintf('%s/figures/connectivity/%dTD_%dASD_granger_%s_%s_%s.png', ...
                path_cluster, length(inds_TD), length(inds_ASD), con, ...
                freq_bin, strjoin(cois, '_')), 'Resolution', 300)
close all

%% behavioral correlations

cois = {'MSS_SWS'};
freq_bin = '18-24Hz'; % freq bin label or 'peak' for finding indiv peak within fmin-fmax
con = 'con_phte-8-AC';
i_tw = 1;
score_labels = {'age'};%, 'ASPSa1-a5', 'ICSS', 'VIQ', 'ADOS_tot_old', 'SRS_tot_T'};
exclude = {}; %'052902', '090902', '048102', '075401', '096603'};
do_plot = 1;
save_brains = 0;
fmin = 8;
fmax = 100;
fmin_plot = 6;
fmax_plot = 60;
peak = 0;

c_TD = [0.56 0.93 0.56];
c_ASD = [0.86 0.44 0.84];

inds_exclude = sort(cellfun(@(x) find(strcmp(sub_IDs, x)), exclude));

% i_coi = find(strcmp(conds, coi));
if freq_bin & ~strcmp(freq_bin, 'peak')
    i_freq_bin = find(strcmp(freq_bin_labels, freq_bin));
end
i_con = find(strcmp(con_labels, con));

for i_score_label=1:length(score_labels)
    score_label = score_labels{i_score_label};
    for i_coi=1:length(cois)
        i_cond = find(strcmp(conds, cois{i_coi}));

        if freq_bin & ~strcmp(freq_bin, 'peak')
            this_data = con_data_binned;
        elseif freq_bin & strcmp(freq_bin, 'peak')
            this_data = con_data;
        else
            this_data = con_data(:,:,:,:,find(foi==fmin_plot):find(foi==fmax_plot));
        end
        this_data(inds_exclude,:,:,:,:) = [];
        this_group_IDs = group_IDs;
        this_group_IDs(inds_exclude) = [];
        
        inds_TD = find(strcmp(this_group_IDs, 'TD'));
        inds_ASD = find(strcmp(this_group_IDs, 'ASD'));
        n_TD = length(inds_TD);
        n_ASD = length(inds_ASD);

        % read scores
        f = sprintf('%s/behavs/%s.txt', path_cluster, score_label);
        fid = fopen(f, 'r');
        scores = [];
        while ~feof(fid)
            scores{end+1,1} = fgetl(fid);
        end
        fclose(fid);
        scores(inds_exclude) = [];
        scores = cellfun(@str2num, scores);
        
        if freq_bin & ~strcmp(freq_bin, 'peak')
            brains = squeeze(this_data(:, i_cond, i_tw, i_con, i_freq_bin));
        else
            brains = squeeze(this_data(:, i_cond, i_tw, i_con, :));
        end

        if save_brains
            save(sprintf('%s/npy/granger_N%d_%.1f-%.1fs_%s_%s_%s.mat', ...
                        path_cluster, length(brains), toi(i_tw), ...
                        toi(i_tw+1), freq_bin, cois{i_coi}, con), 'brains')
        end

        inds_null = find(isnan(scores));
        if freq_bin & ~strcmp(freq_bin, 'peak')
            brains(inds_null) = [];
        else
            brains(inds_null, :) = [];
        end
        scores(inds_null) = [];
        
        this_group_IDs = group_IDs;
        this_group_IDs(inds_exclude) = [];
        this_group_IDs(inds_null) = [];
        inds_TD = find(strcmp(this_group_IDs, 'TD'));
        inds_ASD = find(strcmp(this_group_IDs, 'ASD'));

        if inds_TD
            if freq_bin & ~strcmp(freq_bin, 'peak')
                [r_TD, p_TD] = corr(brains(inds_TD), scores(inds_TD));
                fprintf('\nCorrelation %s %s %.1f-%.1fs %s with %s in TDs: r=%.2f, p=%.3f\n', ...
                        conds{i_cond}, con, toi(i_tw), toi(i_tw+1), freq_bin, score_label, r_TD, p_TD)
            elseif freq_bin & strcmp(freq_bin, 'peak')
                brains_TD = max(brains(inds_TD, find(foi==fmin):find(foi==fmax)), [], 2);
                [r_TD, p_TD] = corr(brains_TD, scores(inds_TD));
                fprintf('\nCorrelation %s %s %.1f-%.1fs %s with %s in TDs: r=%.2f, p=%.3f\n', ...
                        conds{i_cond}, con, toi(i_tw), toi(i_tw+1), freq_bin, score_label, r_TD, p_TD)
            else
                for i=1:size(brains,2)
                    [r_TD(i), p_TD(i)] = corr(brains(inds_TD, i), scores(inds_TD));
                end
            end
        end
        if freq_bin & ~strcmp(freq_bin, 'peak')
            [r_ASD,p_ASD] = corr(brains(inds_ASD), scores(inds_ASD));
            fprintf('\nCorrelation %s %s %.1f-%.1fs %s with %s in ASDs: r=%.2f, p=%.3f\n', ...
                    conds{i_cond}, con, toi(i_tw), toi(i_tw+1), freq_bin, score_label, r_ASD, p_ASD)
        elseif freq_bin & strcmp(freq_bin, 'peak')
            brains_ASD = max(brains(inds_ASD, find(foi==fmin):find(foi==fmax)), [], 2);
            [r_ASD, p_ASD] = corr(brains_ASD, scores(inds_ASD));
            fprintf('\nCorrelation %s %s %.1f-%.1fs %s with %s in ASDs: r=%.2f, p=%.3f\n', ...
                    conds{i_cond}, con, toi(i_tw), toi(i_tw+1), freq_bin, score_label, r_ASD, p_ASD)
        else
            for i=1:size(brains,2)
                    [r_ASD(i), p_ASD(i)] = corr(brains(inds_ASD, i), scores(inds_ASD));
            end
        end

        % difference between correlations here

        if freq_bin & do_plot% & (p_TD<0.07 || p_ASD<0.07)
    %         tiledlayout(1,1,'TileSpacing', 'none', 'Padding', 'normal');
            scatter(scores(inds_TD), brains(inds_TD), 100, c_TD, 'filled');
            hold on
            scatter(scores(inds_ASD), brains(inds_ASD), 100, c_ASD, 'filled');
            h = lsline;
            set(h(1), 'color', c_ASD, 'linewidth', 2)        
            set(h(2), 'color', c_TD, 'linewidth', 2)

            xlims = xlim;
            ylims = ylim;
            text(xlims(1)+xlims(1)*0.08, ylims(2)-(abs(ylims(1))+ylims(2))*0.05, ...
                sprintf('r=%.2f, p=%.3f', r_TD, p_TD), 'color', c_TD)
            text(xlims(1)+xlims(1)*0.08, ylims(2)-(abs(ylims(1))+ylims(2))*0.1, ...
                sprintf('r=%.2f, p=%.3f', r_ASD, p_ASD), 'color', c_ASD)
            xlabel(score_label)
            ylabel('Granger causality')        

            path_save = sprintf('%s/figures/connectivity/correlations/', path_cluster);
            exportgraphics(gcf, sprintf('%s/%dTD_%dASD_granger_%s_%s-%ss_%s_%s_vs_%s.png', ...
                            path_save, length(inds_TD), length(inds_ASD), con, ...
                            num2str(toi(i_tw)), num2str(toi(i_tw+1)), freq_bin, ...
                            conds{i_cond}, score_label), 'Resolution', 300)
            close all
        elseif ~freq_bin & do_plot
            figure
            plot(fmin:fstep:fmax, zscore(mean(brains(inds_TD,:),1)), ...
                'LineWidth', 2)            
            hold on
            plot(fmin:fstep:fmax, zscore(mean(brains(inds_ASD,:),1)), ...
                'LineWidth', 2)
            plot(fmin:fstep:fmax, zscore(r_TD), 'LineWidth', 2, 'Color', c_TD)
            plot(fmin:fstep:fmax, zscore(r_ASD), 'LineWidth', 2, 'Color', c_ASD) 
            axis tight
            xticks(8:4:fmax)
            title(sprintf('%s %.1f-%.1fs', conds{i_cond}, ...
                            toi(i_tw), toi(i_tw+1)));
            xlabel('Frequency (Hz)')
            ylabel('Granger causality')
            if strcmp(conds{i_cond}, 'MSS-SWS')
                loc = 'southwest';
            else
                loc = 'northeast';
            end
            legend('TD', 'ASD', 'Location', loc)
        end
    end
end