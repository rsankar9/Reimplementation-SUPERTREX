% Task 1 - FORCE 
% This script was provided by the Ryan Pyle and Robert Rosenbaum
% Slight modifications were made by Remya Sankar  for ease of use and have been marked
% within the script. No changes were done to the simulation parameters.
%
% This script simulates the FORCE algorithm on Task 1 as presented in
% Pyle, R. and Rosenbaum, R., 2019.
% A reservoir computing model of reward-modulated motor learning and automaticity.
% Neural computation, 31(7), pp.1430-1461.



function y = Task1_FORCE(triallen, res_folder, green, red, orange, purple, grey, rseed) % Alteration #1

    rng(rseed);                                                                         % Alteration #2
    
    
    res_path = res_folder+num2str(rseed);                                               % Alteration #4
    mkdir(res_path);
    

    dt = .2;
    tau = 10;
    Triallen = triallen;                                                                % Alteration #3: 1e4;
    numTrial = 15;                                                                      % 10 train, 5 test
    numTrain = 10;
    alpha = 2.5e-2;
    N = 1000;                                                                           % reservoir size
    p = .1;                                                                             % connection probability
    lambda = 1.5;                                                                       % eigenradius, >1 means unstable
    Jvar = 1/(p*N);                                                                     % standard deviation
    J = sprandn(N,N,p) * lambda * sqrt(Jvar);                                           % connectivity
    J = full(J);
    save(res_path + '/var', 'J');
    
    gamma = 10;
    P = eye(N)/gamma;                                                                   % FORCE inverse correlation estimate initialization

    % Task (Target)
    butt = @(theta) 9 - sin(theta) + 2*sin(3*theta) + 2*sin(5*theta) - sin(7*theta) + 3*cos(2*theta) - 2*cos(4*theta);
    numT = Triallen/dt;
    thetas = linspace(0,2*pi,numT);
    for i = 1:length(thetas)
        xout(i) = butt(thetas(i))*cos(thetas(i))/14.4734;                               % 14.4734 is max r
        yout(i) = butt(thetas(i))*sin(thetas(i))/14.4734;
    end
    NRO = 2;                                                                            % Dimensionality of output
    f = zeros(NRO,numT);                                                                % target function
    f(1,:) = xout;
    f(2,:) = yout;
    f = repmat(f,1,numTrial);

    W_FORCE = zeros(NRO,N);

    %%% Standard Feedback
    Q = rand(N,NRO)*2 - 1;


    x = zeros(N,1);
    x(:) = rand(N,1) - .5*ones(N,1);
    r = tanh(x(:));

    z = zeros(NRO,numT*15);
    W_norm = zeros(1,numT*15);

    for i = 1:1
        %%% Standard Feedback
        x(:) = x(:) + (dt/tau)*(-x(:) +J*r(:) + Q*z(:,i));

        r(:) = tanh(x(:)) + rand(N,1)*alpha*2 - alpha;
        z(:,i) = W_FORCE*r(:);
        zbar = z(:,i);
    end

    for i = 2:Triallen/dt * numTrain
        %%% Standard Feedback
        x(:) = x(:) + (dt/tau)*(-x(:) +J*r(:) + Q*z(:,i-1));

        r(:) = tanh(x(:)) + rand(N,1)*alpha*2 - alpha;
        z(:,i) = W_FORCE*r(:);
        zbar = .8*zbar + .2*z(:,i);

        if mod(i, 10) == 0                                                              % only update every 10 timesteps
            % update inverse correlation matrix
            k = P*r;
            rPr = r'*k;
            c = 1.0/(1.0 + rPr);
            P = P - k*(k'*c);

            % update the error for the linear readout
            e = z(:,i) - f(:,i);
            mse(i) = sum(e.^2);

            % update the output weights
            dw = -e*k'*c;	
            W_FORCE = W_FORCE + dw;
            W_norm(i) = norm(sqrt(W_FORCE*W_FORCE'));
        end
    end
    
    for i = Triallen/dt * numTrain+1:Triallen/dt * numTrial
        %%% Standard + Teacher Forcing
        x(:) = x(:) + (dt/tau)*(-x(:) +J*r(:) + Q*f(:,i-5*Triallen/dt));

        r(:) = tanh(x(:));
        z(:,i) = W_FORCE*r(:);
        e = z(:,i) - f(:,i-numT);
        mse(i) = sum(e.^2);
    end
    
    % Process Output
    zout(:,1) = z(:,1);
    for i = 2:Triallen/dt * numTrial
       zout(:,i) = zout(:,i-1)*.8 + .2*z(:,i); 
    end
    msebar = mse(1);
    for i = 2:Triallen/dt * numTrial
       msebar(i) = msebar(i-1)*.9998 + .0002*mse(i); 
    end
    for i = 2:Triallen/dt * numTrial
        if W_norm(i) == 0
            W_norm(i) = W_norm(i-1);
        end
    end

    mean_sqrtmsebar = mean(sqrt(msebar(Triallen/dt * numTrain:Triallen/dt * numTrial)));
    std_sqrtmsebar = std(sqrt(msebar(Triallen/dt * numTrain:Triallen/dt * numTrial)));
    median_sqrtmsebar = median(sqrt(msebar(Triallen/dt * numTrain:Triallen/dt * numTrial)));
    
    T = table(mean_sqrtmsebar, median_sqrtmsebar, std_sqrtmsebar);
    T.Properties.VariableNames = {'mean', 'median', 'std'};
    writetable(T, res_path + '/test_stats.csv');
%     dlmwrite(res_path + '/Avg_error_in_test_phase.txt', mean_sqrtmsebar);
    
    
    fsize=13;                                                                           % font size
    % Line width
    lwaxes=1.5;                                                                         % axes
    lwcurves=1;                                                                         % plot
    % fig size
    figsize=[200 175];
    datskip = 10;
%     
%     res_path = res_folder+num2str(rseed);                                               % Alteration #4
%     mkdir(res_path);
%     
    figure(1);
    plot(zout(1,:))
    hold on
    plot(f(1,:),'color',red)
    xline(Triallen/dt*numTrain+1, 'Alpha', 0.3, 'LineWidth',2);
    ylim([-1 1]);
    box off
    axis off
    ylabel('x(t)','FontSize',fsize);
    ax = gca;                                                                           % gca = get current axis
    ax.YAxis.Label.Visible='on';
    
    saveas(gcf, res_path + '/CoordinateX.png');                                         % Alteration #5 (all saveas)
    
    figure(2);
    plot(zout(2,:))
    hold on
    plot(f(2,:),'color',red)
    xline(Triallen/dt*numTrain+1, 'Alpha', 0.3, 'LineWidth',2);
    ylim([-1 1]);
    ylabel('y(t)','FontSize',fsize);
    ax = gca;                                                                           % gca = get current axis
    ax.YAxis.Label.Visible='on';
    box off
    axis off
    
    saveas(gcf, res_path + '/CoordinateY.png');

    figure(3);
    plot(zout(1,Triallen/dt*numTrain+1:datskip:Triallen/dt*numTrial),zout(2,Triallen/dt*numTrain+1:datskip:Triallen/dt*numTrial),'LineWidth',lwcurves)
    hold on
    plot(f(1,Triallen/dt*numTrain+1:datskip:Triallen/dt*numTrial),f(2,Triallen/dt*numTrain+1:datskip:Triallen/dt*numTrial),'color',red,'LineWidth',lwcurves)
    set(gca,'YTick',[-1 0 1],'FontSize',fsize)
    set(gca,'XTick',[-1 0 1],'FontSize',fsize)
    set(gca,'LineWidth',lwaxes)
%     title('Drawn Output','FontSize',fsize)
    p=get(gcf,'Position');
    p(3:4)=[400 400];
    set(gcf,'Position',p)
    ylim([-1 1]);
    box off
    axis off
    
    saveas(gcf, res_path + '/TimeSeries.png');

    figure(4);
    semilogy((1:datskip:Triallen/dt*numTrial)*(dt/tau),sqrt(msebar(1:datskip:Triallen/dt*numTrial)))
    xline((Triallen/dt*numTrain+1)*(dt/tau), 'Alpha', 0.3, 'LineWidth', 2);
    line([Triallen/dt*numTrain*(dt/tau) Triallen/dt*numTrial*(dt/tau)], [mean_sqrtmsebar mean_sqrtmsebar], 'LineWidth', 1, 'color', [0.85 0.85 0.85]);
    text(Triallen/dt*numTrial*(dt/tau), mean_sqrtmsebar, num2str(mean_sqrtmsebar, '%.2e'), 'color', [0.85 0.85 0.85], 'FontSize', 13);
%     axis([0 Triallen/dt*numTrial*(dt/tau) 1e-7 3])
    set(gca,'YTick',[0 1e-6 1e-4 1e-2 1],'FontSize',fsize) % manual
%     set(gca,'XTick',[0 Triallen/dt*numTrain Triallen/dt*numTrial]*(dt/tau),'FontSize',fsize)
    set(gca,'LineWidth',lwaxes)
%     xlabel('Time','FontSize',fsize)
    ylabel('Distance from Target','FontSize',fsize)
    p=get(gcf,'Position');
%     p(3:4)=[400 200];
%     set(gcf,'Position',p)
    ax = gca;                                                                           % gca = get current axis
    ax.XAxis.Visible = 'off'; 
    set(gca,'YTick',[0 1e-6 1e-4 1e-2 1],'FontSize',fsize) ;
    ylim([0 1]);
    box off
   
    saveas(gcf, res_path + '/MSE.png');
    
    figure(5);
    plot(W_norm,'color',grey);
    xline(Triallen/dt*numTrain+1, 'Alpha', 0.3, 'LineWidth',2);
    ylabel('||W||','FontSize',fsize);
    ax = gca;                                                                           % gca = get current axis
    ax.XAxis.Visible = 'off'; 
    ylim([0 0.5]);
    set(gca,'YTick',[0 0.5],'FontSize',fsize); % manual
    box off

    saveas(gcf, res_path + '/W_norm.png');
    
    % Alteration #6: Creating overall figure
    figure(6)
    ax = zeros(5,1);
    for i = 1:5
        ax(i)=subplot(5,1,i);
    end
    
    for i = 1:5
        figure(i)
        h = get(gcf,'Children');
        newh = copyobj(h,6);
        for j = 1:length(newh)
            posnewh = get(newh(j),'Position');
            possub  = get(ax(i),'Position');
            set(newh(j),'Position',...
            [posnewh(1) possub(2) posnewh(3) possub(4)])
        end
        delete(ax(i));
    end
    figure(6)
    
    saveas(gcf, res_path + '/Overall.png');

end
