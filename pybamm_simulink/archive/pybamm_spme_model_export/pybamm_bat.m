classdef pybamm_bat < matlab.System %& matlab.system.mixin.Propagates
    % Simulate a li-ion battery with pybamm
    %
    % This template includes the minimum set of functions required
    % to define a System object with discrete state.

    % Public, tunable properties
    
    properties
        %
    end

    properties(DiscreteState)
        %
    end

    % Pre-computed constants
    properties(Access = private)
        y0
        f
        variables
        first_step
    end
    
    methods(Access = protected)
        %% SET UP SIMULINK SYSTEM INPUTS AND OUTPUTS
        function num = getNumInputsImpl(~)
            num = 1;
        end
        function num = getNumOutputsImpl(~)
            num = 9;
        end
        function [dt1, dt2, dt3, dt4, dt5, dt6, dt7, dt8, dt9] = getOutputDataTypeImpl(~)
        	dt1 = 'double';
            dt2 = 'double';
            dt3 = 'double';
            dt4 = 'double';
            dt5 = 'double';
            dt6 = 'double';
            dt7 = 'double';
            dt8 = 'double';
            dt9 = 'double';
        end
        function [dt1] = getInputDataTypeImpl(~)
        	dt1 = 'double';
%             dt2 = 'double';
%             dt3 = 'double';
        end
        function [sz1, sz2, sz3, sz4, sz5, sz6, sz7, sz8, sz9] = getOutputSizeImpl(~)
        	sz1 = 1;
            sz2 = 1;
            sz3 = 1;
            sz4 = 60;
            sz5 = 1;
            sz6 = 20;
            sz7 = 20;
            sz8 = 1;
            sz9 = 1;
        end
        function [sz1] = getInputSizeImpl(~)
        	sz1 = 1;
%             sz2 = 1;
%             sz3 = 1;
        end
        function [cp1] = isInputComplexImpl(~)
        	cp1 = false;
%             cp2 = false;
%             cp3 = false;
        end
        function [cp1, cp2, cp3, cp4, cp5, cp6, cp7, cp8, cp9] = isOutputComplexImpl(~)
        	cp1 = false;
            cp2 = false;
            cp3 = false;
            cp4 = false;
            cp5 = false;
            cp6 = false;
            cp7 = false;
            cp8 = false;
            cp9 = false;
        end
        function [fz1] = isInputFixedSizeImpl(~)
        	fz1 = true;
%             fz2 = true;
%             fz3 = true;
        end
        function [fz1, fz2, fz3, fz4, fz5, fz6, fz7, fz8, fz9] = isOutputFixedSizeImpl(~)
        	fz1 = true;
            fz2 = true;
            fz3 = true;
            fz4 = true;
            fz5 = true;
            fz6 = true;
            fz7 = true;
            fz8 = true;
            fz9 = true;
        end
        %% LOAD CASADI OBJECTS EXPORTED FROM PYTHON (./TEMP)
        function setupImpl(obj)
            % Perform one-time calculations, such as computing constants
            import casadi.*
            %tmp = load('temp\y0.mat');
            %obj.y0 = inf;
            obj.f = Function.load('./temp/integrator.casadi');
            obj.variables = Function.load('./temp/variables.casadi');
            obj.first_step = true;
        end

        %%
        function [V_t, ocv, T, ce, heating, neg_conc, pos_conc, x_sei, V_meas] = stepImpl(obj, current)
            % Implement algorithm. Calculate y as a function of input u and
            % discrete states.

            % Define initial states from t=0 from python-pybamm 
            if obj.first_step 
%                 if soc_init == 1.0 % not fully charged. just used as a case selector 
                tmp_top = load('temp\x0.mat'); % actually set              
                obj.y0 = tmp_top.x0;
%                 else
%                     tmp_bot = load('temp\y0_bot.mat'); 
%                     obj.y0 = tmp_bot.y0;
%                 end
                obj.first_step = false;
            end
%             T_ref = 298.15;
%             Delta_T = 1.0;
%             T_non_dim = (temperature - T_ref) / Delta_T;
%             T_min = 1e-6;
%             inputs = [T_non_dim, current];
            yt = obj.f(obj.y0, current, 0, 0, 0, 0); %horzcat(inputs, T_min)
            obj.y0 = yt(:, end);
            

            % Define model outputs 
            
            temp = double(full(obj.variables(0, obj.y0, 0, current)));
            V_t = temp(1);
            ocv = temp(2);
            T = temp(3);
            ce = temp(4:(4+60-1)); % 30/electrode
            heating = temp(64);
            neg_conc = temp(65:(65+20-1));
            pos_conc = temp(85:(85+20-1));
            x_sei = temp(105);
            V_meas = temp(106);
%             internal_resistance = abs((ocp - v)/current);
        end

        function resetImpl(obj)
            % Initialize / reset discrete-state properties
        end
    end
end