% Task 3 - SUPERTREX - Beta 0.01 
% This script was provided by the Ryan Pyle and Robert Rosenbaum
% Slight modifications were made by Remya Sankar  for ease of use and have been marked
% within the script. No changes were done to the simulation parameters.
%
% This script simulates the SUPERTREX algorithm on a variant of Task 3 as presented in
% Pyle, R. and Rosenbaum, R., 2019.
% A reservoir computing model of reward-modulated motor learning and automaticity.
% Neural computation, 31(7), pp.1430-1461.
% Cost:- [0.1, 0.01, 0.0]


function y = Task3_ST_beta(triallen, res_folder, green, red, orange, purple, grey, rseed) % Alteration #1

    rng(rseed);                                                                                    % Alteration #2
    
    res_path = res_folder+num2str(rseed);                                               % Alteration #4
    mkdir(res_path);                                                              % Alteration #2

    alpha2 = .1*1; % additional cost of moving joint 1
    beta2 = .01*1;  % additional cost of moving joint 2
    gamma2 = 0;  % additional cost of moving joint 3

    dt = .2;
    tau = 10;
    Triallen = triallen;                                                                % Alteration #3: 1e4;
    numTrial = 15; % 10 train, 5 test
    numTrain = 10;
    alpha = 2.5e-2;
    beta = 1e-2;
    N = 1000; % reservoir size
    p = .1; % connection probability
    lambda = 1.5; % eigenradius, >1 means unstable
    Jvar = 1/(p*N); % standard deviation
    J = sprandn(N,N,p) * lambda * sqrt(Jvar); % connectivity
    J = full(J);
    save(res_path + '/var', 'J');
    gamma = 10;
    P = eye(N)/gamma; % FORCE inverse correlation estimate initialization

    % Task (Target)
    butt = @(theta) 9 - sin(theta) + 2*sin(3*theta) + 2*sin(5*theta) - sin(7*theta) + 3*cos(2*theta) - 2*cos(4*theta);
    numT = Triallen/dt;
    thetas = linspace(0,2*pi,numT);
    for i = 1:length(thetas)
        xout(i) = butt(thetas(i))*cos(thetas(i))/14.4734; % 14.4734 is max r
        yout(i) = butt(thetas(i))*sin(thetas(i))/14.4734;
        sout(i) = i/length(thetas);
        sout2(i) = cos(thetas(i));
        sout3(i) = sin(thetas(i));
    end
    NRO = 3; % Dimensionality of output
    f = zeros(2,numT); % target function
    f(1,:) = xout;
    f(2,:) = yout;
    f = repmat(f,1,numTrial);
    % Double State Feedback (circle)
    fs = zeros(2,numT); % target function
    fs(1,:) = sout2;
    fs(2,:) = sout3;
    fs = repmat(fs,1,numTrial);
    % Position of joints from output (angles)
    Poseqn = @(x) [0+ sin(x(1)*pi)*1.8 + sin((x(1)+x(2))*pi)*1.2 + sin((x(1)+x(2)+x(3))*pi)*.6;
               -2 + cos(x(1)*pi)*1.8 + cos((x(1)+x(2))*pi)*1.2 + cos((x(1)+x(2)+x(3))*pi)*.6];
    Pos1eqn = @(x) [0 + sin(x(1)*pi)*1.8;
               -2 + cos(x(1)*pi)*1.8]; % position of first pivot
    Pos2eqn = @(x) [0 + sin(x(1)*pi)*1.8 + sin((x(1)+x(2))*pi)*1.2;
               -2 + cos(x(1)*pi)*1.8 + cos((x(1)+x(2))*pi)*1.2]; % position of second pivot

    W_RMHL = zeros(NRO,N); 
    W_FORCE = zeros(NRO,N); 

    %%% Standard Feedback
    Q = rand(N,NRO)*2 - 1;

    x = zeros(N,1);
    x(:) = rand(N,1) - .5*ones(N,1);
    r = tanh(x(:));

    z = zeros(NRO,numT*15);
    z_exp = zeros(NRO,numT*15);
    z_mst = zeros(NRO,numT*15);
    W_exp_norm = zeros(1,numT*15);
    W_mst_norm = zeros(1,numT*15);

    betause = beta;

    learningrate = .0005;

    for i = 1:1
        %%% Standard Feedback
        x(:) = x(:) + (dt/tau)*(-x(:) +J*r(:) + Q*z(:,i));
        r(:) = tanh(x(:)) + rand(N,1)*alpha*2 - alpha;
        z_exp(:,i) = W_RMHL*r(:) + rand(NRO,1)*2*betause - betause;
        z_mst(:,i) = W_FORCE*r(:);
        z(:,i) = z_exp(:,i) + z_mst(:,i);
        zbar(:,1) = z(:,i);

        Pos = Poseqn(z(:,i));

        mse(i) = sum((Pos-f(:,i)).^2);
        mse(i) = mse(i) + alpha2*abs(z(1,i)-zbar(1,i)) + beta2*abs(z(2,i)-zbar(2,i)) + gamma2*abs(z(3,i)-zbar(3,i));
        mse_delta = mse(i);
    end

    for i = 2:Triallen/dt * numTrain
        %%% Standard Feedback
        x(:) = x(:) + (dt/tau)*(-x(:) +J*r(:) + Q*z(:,i-1));

        r(:) = tanh(x(:)) + rand(N,1)*alpha*2 - alpha;
        z_exp(:,i) = W_RMHL*r(:) + rand(NRO,1)*2*betause - betause;
        z_mst(:,i) = W_FORCE*r(:);
        z(:,i) = z_exp(:,i) + z_mst(:,i);
        zbar(:,i) = .8*zbar(:,i-1) + .2*z_exp(:,i);

        Pos = Poseqn(z(:,i));

        mse(i) = sum((Pos-f(:,i)).^2);
        mse(i) = mse(i) + alpha2*abs(z(1,i)-zbar(1,i)) + beta2*abs(z(2,i)-zbar(2,i)) + gamma2*abs(z(3,i)-zbar(3,i));
        mse_delta = .8*mse_delta + .2*mse(i);

        betause = .005*(10*mse_delta)^.25; % Task 3

        test = -(mse(i)-mse_delta);
        if (test >= 0)
            testuse = -(abs(test).^.25);
            W_RMHL = W_RMHL + .5*learningrate*(-5*testuse)*(z_exp(:,i) - zbar(:,i))*r(:)';
        else
            testuse = -(abs(test).^.25);
            W_RMHL = W_RMHL + .5*learningrate*(5*testuse)*(z_exp(:,i) - zbar(:,i))*r(:)';
        end

        W_exp_norm(i) = norm(sqrt(W_RMHL*W_RMHL'));

        % FORCE (TOP)
        if (mod(i, 10) == 0)
            % update inverse correlation matrix
            k = P*r;
            rPr = r'*k;
            c = 1.0/(1.0 + rPr);
            %P = P - k*(k'*c);
            P = P - k*(k'*c)*(-.5*tanh(500000*(abs(mse_delta)-(15.e-3)))+.5);

            % update the error for the linear readout
            e = -z_exp(:,i);
            e = -zbar(:,i); % smooth noise?

            % update the output weights
            dw = .9*-e*k'*c*(-.5*tanh(500000*(abs(mse_delta)-(15.e-3)))+.5);
            %dw = .5*-e*k'*c;	
            W_FORCE = W_FORCE + dw;
            W_mst_norm(i) = norm(sqrt(W_FORCE*W_FORCE'));
        end
    end
    W_RMHL = 0;
    for i = Triallen/dt * numTrain+1:Triallen/dt * numTrial
        %%% Standard + Teacher Forcing
        x(:) = x(:) + (dt/tau)*(-x(:) +J*r(:) + Q*z(:,i-5*Triallen/dt));


        r(:) = tanh(x(:));
        z(:,i) = W_FORCE*r(:);
        zbar(:,i) = .8*zbar(:,i-1) + .2*z_exp(:,i);
        Pos = Poseqn(z(:,i));
        e = Pos - f(:,i-numT);
        mse(i) = sum(e.^2);
        mse(i) = mse(i) + alpha2*abs(z(1,i)-zbar(1,i)) + beta2*abs(z(2,i)-zbar(2,i)) + gamma2*abs(z(3,i)-zbar(3,i));
    end
    % Process Output
    zout(:,1) = z(:,1);
    Pos = Poseqn(z(:,1));
    Pos1 = Pos1eqn(z(:,1));
    zpos(:,1) = Pos;
    zpos1(:,1) = Pos1;
    for i = 2:Triallen/dt * numTrial
       zout(:,i) = zout(:,i-1)*.8 + .2*z(:,i); 
       Pos = Poseqn(z(:,i));
       Pos1 = Pos1eqn(z(:,i));
       zpos(:,i) = zpos(:,i-1)*.8 + .2*Pos;
       zpos1(:,i) = zpos1(:,i-1)*.8 + .2*Pos1;
    end


    msebar = mse(1);
    for i = 2:Triallen/dt * numTrial
       msebar(i) = msebar(i-1)*.9998 + .0002*mse(i); 
    end
    for i = 2:Triallen/dt * numTrial
        if W_exp_norm(i) == 0
            W_exp_norm(i) = W_exp_norm(i-1);
        end
        if W_mst_norm(i) == 0
            W_mst_norm(i) = W_mst_norm(i-1);
        end
    end

    mean_sqrtmsebar = mean(sqrt(msebar(Triallen/dt * numTrain:Triallen/dt * numTrial)));
    std_sqrtmsebar = std(sqrt(msebar(Triallen/dt * numTrain:Triallen/dt * numTrial)));
    median_sqrtmsebar = median(sqrt(msebar(Triallen/dt * numTrain:Triallen/dt * numTrial)));
    
    T = table(mean_sqrtmsebar, median_sqrtmsebar, std_sqrtmsebar);
    T.Properties.VariableNames = {'mean', 'median', 'std'};
    writetable(T, res_path + '/test_stats.csv');
%     dlmwrite(res_path + '/Avg_error_in_test_phase.txt', mean_sqrtmsebar);
    

    fsize=13; % font size
    % Line width
    lwaxes=1.5; % axes
    lwcurves=1; % plot
    % fig size
    figsize=[200 175];
    datskip = 10;
    
%     res_path = res_folder+num2str(rseed);                                               % Alteration #4
%     mkdir(res_path);

    figure(1);
    plot(zpos(1,:))
    hold on
    plot(f(1,:),'color',red)
    xline(Triallen/dt*numTrain+1, 'Alpha', 0.3, 'LineWidth',2);
    ylim([-1.5 1.5]);
    box off
    axis off
    ylabel('x(t)','FontSize',fsize);
    ax = gca;                   % gca = get current axis
    ax.YAxis.Label.Visible='on';
    
    saveas(gcf, res_path + '/CoordinateX.png');                             % Alteration #5 (all saveas)
    
    figure(2);
    plot(zpos(2,:))
    hold on
    plot(f(2,:),'color',red)
    xline(Triallen/dt*numTrain+1, 'Alpha', 0.3, 'LineWidth',2);
    ylim([-1.5 1.5]);
    ylabel('y(t)','FontSize',fsize);
    ax = gca;                   % gca = get current axis
    ax.YAxis.Label.Visible='on';
    box off
    axis off
    
    saveas(gcf, res_path + '/CoordinateY.png');

    figure(3);
    plot(zpos(1,Triallen/dt*numTrain+1:datskip:Triallen/dt*numTrial),zpos(2,Triallen/dt*numTrain+1:datskip:Triallen/dt*numTrial),'LineWidth',lwcurves)
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
    xline((Triallen/dt*numTrain+1)*(dt/tau), 'Alpha', 0.3, 'LineWidth',2);
    line([Triallen/dt*numTrain*(dt/tau) Triallen/dt*numTrial*(dt/tau)], [mean_sqrtmsebar mean_sqrtmsebar], 'LineWidth', 1, 'color', [0.85 0.85 0.85]);
    text(Triallen/dt*numTrial*(dt/tau), mean_sqrtmsebar, num2str(mean_sqrtmsebar, '%.2e'), 'color', [0.85 0.85 0.85], 'FontSize', 13);
%     axis([0 Triallen/dt*numTrial*(dt/tau) 1e-7 3])
    set(gca,'YTick',[0 1e-6 1e-4 1e-2 1 1e1],'FontSize',fsize) ;
%     set(gca,'XTick',[0 Triallen/dt*numTrain Triallen/dt*numTrial]*(dt/tau),'FontSize',fsize)
    set(gca,'LineWidth',lwaxes)
%     xlabel('Time','FontSize',fsize)
    ylabel('Distance from Target','FontSize',fsize)
    p=get(gcf,'Position');
%     p(3:4)=[400 200];
%     set(gcf,'Position',p)
    ax = gca;                   % gca = get current axis
    ax.XAxis.Visible = 'off'; 
    ylim([0 10]);
    box off
   
    saveas(gcf, res_path + '/MSE.png');
    
    figure(5);
    plot(W_exp_norm,'color',green);
    hold on;
    plot(W_mst_norm,'color',purple);
    xline(Triallen/dt*numTrain+1, 'Alpha', 0.3, 'LineWidth',2);
    ylabel('||W||','FontSize',fsize);
    ax = gca;                   % gca = get current axis
    ax.XAxis.Visible = 'off'; 
    ylim([0 0.5]);
    set(gca,'YTick',[0 0.5],'FontSize',fsize); % manual
    box off

    saveas(gcf, res_path + '/W_norm.png');
    
    figure(6);
    plot(zout(1,:))
    xline(Triallen/dt*numTrain+1, 'Alpha', 0.3, 'LineWidth',2);
    ylim([-1.5 1.5]);
    ylabel('\theta_1','FontSize',fsize);
    ax = gca;                   % gca = get current axis
    ax.YAxis.Label.Visible='on';
    box off
    axis off
    
    saveas(gcf, res_path + '/Theta1.png');
    
    figure(7);
    plot(zout(2,:))
    xline(Triallen/dt*numTrain+1, 'Alpha', 0.3, 'LineWidth',2);
    ylim([-1.5 1.5]);
    ylabel('\theta_2','FontSize',fsize);
    ax = gca;                   % gca = get current axis
    ax.YAxis.Label.Visible='on';
    box off
    axis off
    
    saveas(gcf, res_path + '/Theta2.png');
    
    figure(8);
    plot(zout(3,:))
    xline(Triallen/dt*numTrain+1, 'Alpha', 0.3, 'LineWidth',2);
    ylim([-1.5 1.5]);
    ylabel('\theta_3','FontSize',fsize);
    ax = gca;                   % gca = get current axis
    ax.YAxis.Label.Visible='on';
    box off
    axis off
    
    saveas(gcf, res_path + '/Theta3.png');
    
    % Alteration #6: Creating overall figure
    figure(9)
    ax = zeros(8,1);
    for i = 1:8
        ax(i)=subplot(8,1,i);
    end
    
    for i = 1:8
        figure(i)
        h = get(gcf,'Children');
        newh = copyobj(h,9);
        for j = 1:length(newh)
            posnewh = get(newh(j),'Position');
            possub  = get(ax(i),'Position');
            set(newh(j),'Position',...
            [posnewh(1) possub(2) posnewh(3) possub(4)])
        end
        delete(ax(i));
    end
    figure(9)
    
    saveas(gcf, res_path + '/Overall.png');
    save(res_path + '/Data.mat', 'msebar', 'W_exp_norm', 'W_mst_norm');
    

end