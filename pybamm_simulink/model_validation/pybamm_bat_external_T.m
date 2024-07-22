classdef pybamm_bat_external_T < matlab.System %& matlab.system.mixin.Propagates
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
            num = 2;
        end
        function num = getNumOutputsImpl(~)
            num = 10;
        end
        function [dt1, dt2, dt3, dt4, dt5, dt6, dt7, dt8, dt9, dt10] = getOutputDataTypeImpl(~)
        	dt1 = 'double';
            dt2 = 'double';
            dt3 = 'double';
            dt4 = 'double';
            dt5 = 'double';
            dt6 = 'double';
            dt7 = 'double';
            dt8 = 'double';
            dt9 = 'double';
            dt10 = 'double';
            % dt11 = 'double';
        end
        function [dt1] = getInputDataTypeImpl(~)
        	dt1 = 'double';
%             dt2 = 'double';
%             dt3 = 'double';
        end
        function [sz1, sz2, sz4, sz5, sz6, sz7, sz8, sz9,sz10, sz11] = getOutputSizeImpl(~)
        	sz1 = 1;
            sz2 = 1;
            % sz3 = 1;
            sz4 = 60;
            sz5 = 1;
            sz6 = 20;
            sz7 = 20;
            sz8 = 1;
            sz9 = 1;
            sz10 = 20;
            sz11 = 20;
        end
        function [sz1, sz2] = getInputSizeImpl(~)
        	sz1 = 1;
            sz2 = 1;
%             sz3 = 1;
        end
        function [cp1,cp2] = isInputComplexImpl(~)
        	cp1 = false;
            cp2 = false;
%             cp3 = false;
        end
        function [cp1, cp2, cp3, cp4, cp5, cp6, cp7, cp8, cp9, cp10] = isOutputComplexImpl(~)
        	cp1 = false;
            cp2 = false;
            cp3 = false;
            cp4 = false;
            cp5 = false;
            cp6 = false;
            cp7 = false;
            cp8 = false;
            cp9 = false;
            cp10 = false;
            % cp11 = false;
        end
        function [fz1, fz2] = isInputFixedSizeImpl(~)
        	fz1 = true;
            fz2 = true;
%             fz3 = true;
        end
        function [fz1, fz2, fz3, fz4, fz5, fz6, fz7, fz8, fz9, fz10] = isOutputFixedSizeImpl(~)
        	fz1 = true;
            fz2 = true;
            fz3 = true;
            fz4 = true;
            fz5 = true;
            fz6 = true;
            fz7 = true;
            fz8 = true;
            fz9 = true;
            fz10 = true;
            % fz11 = true;
        end
        %% LOAD CASADI OBJECTS EXPORTED FROM PYTHON (./TEMP)
        function setupImpl(obj)
            % Perform one-time calculations, such as computing constants
            import casadi.*
            %tmp = load('temp\y0.mat');
            %obj.y0 = inf;
            obj.f = Function.load('./temp_extT/integrator.casadi');
            obj.variables = Function.load('./temp_extT/variables.casadi');
            obj.first_step = true;
        end

        %%
        function [V_t, ocv, ce, heating, neg_conc, pos_conc, x_sei, V_meas, ocp_n, ocp_p] = stepImpl(obj, I, T)
            % Implement algorithm. Calculate y as a function of input u and
            % discrete states.

            % Define initial states from t=0 from python-pybamm 
            if obj.first_step 
%                 if soc_init == 1.0 % not fully charged. just used as a case selector 
                tmp_top = load('temp_extT\x0.mat'); % actually set              
                obj.y0 = tmp_top.x0;
%                 else
%                     tmp_bot = load('temp\y0_bot.mat'); 
%                     obj.y0 = tmp_bot.y0;
%                 end
                obj.first_step = false;
            end
            T_ref = 298.15;
            Delta_T = 1.0;
            T_non_dim = (T - T_ref) / Delta_T;
            T_min = 1e-6;
            inputs = [T_non_dim, I];
            yt = obj.f(obj.y0, horzcat(inputs, T_min), 0, 0, 0, 0);
            obj.y0 = yt(:, end);
            

            % Define model outputs 
            temp = double(full(obj.variables(0, obj.y0, 0, I)));
            V_t = temp(1);
            ocv = temp(2);
            ce = temp(3:(3+60-1)); % 30/electrode
            heating = temp(63);
            neg_conc = temp(64:(64+20-1));
            pos_conc = temp(84:(84+20-1));
            x_sei = temp(104);
            V_meas = temp(105);
            ocp_n = temp(106:(106+20-1));
            ocp_p = temp(126:(126+20-1));
%             internal_resistance = abs((ocp - v)/current);
        end

        function resetImpl(obj)
            % Initialize / reset discrete-state properties
        end
    end
end