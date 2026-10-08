classdef TestMATLINK < matlab.unittest.TestCase
    % Real Studio broker requests executed through the MATLAB event-loop bridge.
    properties
        Project
    end
    methods (TestClassSetup)
        function environment(testCase)
            executable = string(getenv('OPENDPD_MATLAB_PYTHON'));
            testCase.assertNotEmpty(char(executable));
            opendpd.setup(PythonExecutable=executable);
            testCase.Project = opendpd.openProject(string(tempname) + " MATLINK workspace");
            testCase.addTeardown(@() opendpd.closeProject(testCase.Project, StopService=true));
        end
    end
    methods (Test)
        function connectionReuseAndDisconnect(testCase)
            p = testCase.Project;
            link = opendpd.studio(p, OpenBrowser=false, Label="MATLINK integration test");
            testCase.addTeardown(@() delete(link));
            same = opendpd.studio(p, OpenBrowser=false);
            testCase.verifyTrue(link == same);
            state = testCase.api('GET', '/matlink');
            connection = state.sessions(strcmp({state.sessions.client_id}, char(link.ClientID)));
            testCase.verifyTrue(connection.connected);
            request = testCase.request(link, 'create_demo', struct(), 'disconnect-demo');
            id = char(link.ClientID);
            opendpd.disconnect(link);
            state = testCase.api('GET', '/matlink');
            connection = state.sessions(strcmp({state.sessions.client_id}, id));
            testCase.verifyFalse(connection.connected);
            transfer = state.transfers(strcmp({state.transfers.request_id}, request.request_id));
            testCase.verifyEqual(transfer.status, 'failed');
            info = p.Backend.studio_info();
            testCase.verifyTrue(logical(info.get('ready')) || ~isempty(info.get('problems')));
        end

        function signalsAndAutomaticReportDelivery(testCase)
            link = opendpd.studio(testCase.Project, OpenBrowser=false);
            testCase.addTeardown(@() delete(link));
            originalRNG = rng;
            request = testCase.request(link, 'create_demo', struct(), 'demo-once');
            link.poll();
            testCase.assertTrue(link.Connected, char(link.LastError));
            transfer = testCase.transfer(request.request_id);
            testCase.assertEqual(transfer.status, 'succeeded', transfer.error);
            fields = transfer.result;
            testCase.addTeardown(@() evalin('base', sprintf('clear %s %s', fields.input, fields.output)));
            testCase.verifyEqual(rng, originalRNG);
            testCase.verifyEqual(numel(evalin('base', fields.input)), 4096);
            link.poll();
            state = testCase.api('GET', '/matlink');
            session = state.sessions(strcmp({state.sessions.client_id}, char(link.ClientID)));
            variable = session.variables(strcmp({session.variables.name}, fields.input));
            testCase.verifyTrue(variable.eligible);
            testCase.verifyEqual(variable.n_samples, 4096);
            payload = rmfield(fields, 'n_samples');
            [~, suffix] = fileparts(tempname);
            payload.name = ['matlink-' suffix(1:10)];
            request = testCase.request(link, 'import_iq', payload, 'iq-once');
            link.poll();
            transfer = testCase.transfer(request.request_id);
            testCase.assertEqual(transfer.status, 'succeeded', transfer.error);
            dataset = transfer.result.dataset_id;
            training = struct('epochs', 2, 'frame_length', 32, 'frame_stride', 32, ...
                'batch_size', 16, 'batch_size_eval', 16);
            job = opendpd.trainPA(testCase.Project, string(dataset), ...
                ModelParameters=struct('hidden_size', 4), Training=training);
            sentinel = matlab.lang.makeValidName(['pa_gru_' dataset]);
            testCase.assertFalse(any(strcmp(evalin('base', 'who'), sentinel)));
            assignin('base', sentinel, single(123));
            testCase.addTeardown(@() evalin('base', ['clear ' sentinel]));
            request = testCase.request(link, 'import_result', struct('run_id', char(job.ID)), 'report-once');
            testCase.verifyTrue(any(strcmp(request.status, {'waiting', 'queued'})));
            opendpd.wait(job);
            link.poll();
            transfer = testCase.transfer(request.request_id);
            testCase.assertEqual(transfer.status, 'succeeded', transfer.error);
            name = transfer.result.variable;
            testCase.addTeardown(@() evalin('base', ['clear ' name]));
            testCase.verifyNotEqual(name, sentinel);
            testCase.verifyEqual(evalin('base', sentinel), single(123));
            testCase.verifyTrue(isstruct(evalin('base', name)));
            % Repeated clicks with a fresh browser request reuse the imported report.
            another = testCase.request(link, 'import_result', struct('run_id', char(job.ID)), 'report-again');
            link.poll();
            second = testCase.transfer(another.request_id);
            testCase.verifyEqual(second.result.variable, name);
            assignin('base', name, single(321));
            another = testCase.request(link, 'import_result', struct('run_id', char(job.ID)), 'report-after-edit');
            link.poll();
            third = testCase.transfer(another.request_id);
            thirdName = third.result.variable;
            testCase.addTeardown(@() evalin('base', ['clear ' thirdName]));
            testCase.verifyNotEqual(thirdName, name);
            testCase.verifyEqual(evalin('base', name), single(321));
            testCase.verifyTrue(isstruct(evalin('base', thirdName)));
            bundled = testCase.request(link, 'import_result', struct('run_id', char(job.ID), ...
                'variable', sentinel, 'bundle', true), 'report-bundle');
            link.poll();
            bundled = testCase.transfer(bundled.request_id);
            testCase.assertEqual(bundled.status, 'succeeded', bundled.error);
            bundleName = bundled.result.variable;
            testCase.addTeardown(@() evalin('base', ['clear ' bundleName]));
            testCase.verifyNotEqual(bundleName, sentinel);
            stored = evalin('base', bundleName);
            testCase.verifyTrue(isfield(stored, 'configuration'));
            testCase.verifyTrue(isfield(stored.plots, 'spectrum'));
            testCase.verifyEqual(stored.run_id, char(job.ID));
            testCase.verifyEqual(evalin('base', sentinel), single(123));
            url = opendpd.openStudio(testCase.Project, Page="matlink", OpenBrowser=false);
            testCase.verifyTrue(contains(url, 'next=%2Fmatlink'));
        end

        function ownedConnectionLeavesServiceAlive(testCase)
            workspace = string(tempname) + " owned MATLINK";
            link = opendpd.studio(workspace, OpenBrowser=false);
            testCase.addTeardown(@() delete(link));
            owned = link.Project;
            testCase.addTeardown(@() opendpd.closeProject(owned, StopService=true));
            opendpd.disconnect(workspace);
            resumed = opendpd.openProject(workspace, StartService=false);
            testCase.addTeardown(@() opendpd.closeProject(resumed));
            testCase.verifyEqual(resumed.Workspace, owned.Workspace);
        end

        function generatedCollectionSavedAsComplexVectors(testCase)
            link = opendpd.studio(testCase.Project, OpenBrowser=false);
            testCase.addTeardown(@() delete(link));
            configs = {struct('preset_id', 'matlab-tone-a', 'waveform', 'tone', ...
                'n_samples', 8192, 'sample_rate_hz', 80e6), ...
                struct('preset_id', 'matlab-tone-b', 'waveform', 'tone', ...
                'n_samples', 8192, 'sample_rate_hz', 80e6, 'seed', 43)};
            signals = testCase.api('POST', '/signal-generator/batches', struct('configs', {configs}));
            paired = testCase.api('POST', '/pa-library/datasets', struct('input_signal_ids', {{signals.signal_id}}, ...
                'model_id', 'rapp-am-pm', 'parameters', struct()));
            datasetID = paired.dataset.dataset_id;
            request = testCase.request(link, 'import_dataset', struct('dataset_id', datasetID), 'generated-collection');
            link.poll();
            completed = testCase.transfer(request.request_id);
            testCase.assertEqual(completed.status, 'succeeded', completed.error);
            name = completed.result.variable;
            testCase.addTeardown(@() evalin('base', ['clear ' name]));
            saved = evalin('base', name);
            testCase.verifyEqual(saved.dataset_id, datasetID);
            testCase.verifyNumElements(saved.captures, 2);
            for index = 1:2
                testCase.verifySize(saved.captures(index).x, [8192 1]);
                testCase.verifyClass(saved.captures(index).x, 'single');
                testCase.verifyFalse(isreal(saved.captures(index).x));
                testCase.verifySize(saved.captures(index).y, [8192 1]);
                testCase.verifyEqual(saved.captures(index).signal.sample_rate_hz, configs{index}.sample_rate_hz);
            end
            again = testCase.request(link, 'import_dataset', struct('dataset_id', datasetID), 'generated-again');
            link.poll();
            again = testCase.transfer(again.request_id);
            testCase.verifyEqual(again.result.variable, name);
            assignin('base', name, 123);
            again = testCase.request(link, 'import_dataset', struct('dataset_id', datasetID), 'generated-after-edit');
            link.poll();
            again = testCase.transfer(again.request_id);
            testCase.addTeardown(@() evalin('base', ['clear ' again.result.variable]));
            testCase.verifyNotEqual(again.result.variable, name);
            testCase.verifyEqual(evalin('base', name), 123);
        end

        function lostAcknowledgmentDoesNotRepeatOperation(testCase)
            link = opendpd.studio(testCase.Project, OpenBrowser=false);
            testCase.addTeardown(@() delete(link));
            module = py.importlib.import_module('opendpd.sdk.matlink');
            original = py.getattr(module.MatlinkClient, '_request');
            calls = py.list();
            context = py.dict(pyargs('original', original, 'calls', calls));
            script = strjoin(["def lost_ack(self, suffix, body):", ...
                "    result = original(self, suffix, body)", ...
                "    if suffix.endswith('/complete'):", ...
                "        calls.append(suffix)", ...
                "        raise RuntimeError('simulated lost acknowledgment')", ...
                "    return result"], newline);
            py.exec(char(script), context);
            faulty = context.get('lost_ack');
            py.setattr(module.MatlinkClient, '_request', faulty);
            testCase.addTeardown(@() py.setattr(module.MatlinkClient, '_request', original));
            request = testCase.request(link, 'create_demo', struct(), 'demo-lost-ack');
            before = evalin('base', 'who');
            link.poll();
            transfer = testCase.transfer(request.request_id);
            testCase.assertEqual(transfer.status, 'succeeded');
            fields = transfer.result;
            testCase.addTeardown(@() evalin('base', sprintf('clear %s %s', fields.input, fields.output)));
            after = evalin('base', 'who');
            testCase.verifyEqual(numel(setdiff(after, before)), 2);
            testCase.verifyGreaterThan(double(py.len(calls)), 0);
            py.setattr(module.MatlinkClient, '_request', original);
            link.poll();
            testCase.verifyEqual(evalin('base', 'who'), after);
            testCase.verifyTrue(link.Connected);
            testCase.verifyEmpty(char(link.LastError));
        end
    end
    methods (Access = private)
        function value = api(testCase, method, path, body)
            request = py.getattr(testCase.Project.Backend, '_request');
            if nargin < 4
                response = request(method, path);
            else
                response = request(method, path, py.json.loads(jsonencode(body)));
            end
            module = py.importlib.import_module('opendpd.sdk.matlab');
            value = jsondecode(char(module.encode(response)));
        end

        function value = request(testCase, link, action, payload, key)
            value = testCase.api('POST', '/matlink/requests', struct('client_id', char(link.ClientID), ...
                'action', action, 'payload', payload, 'idempotency_key', key));
        end

        function value = transfer(testCase, id)
            state = testCase.api('GET', '/matlink');
            value = state.transfers(strcmp({state.transfers.request_id}, id));
        end
    end
end
