% Part A of the feasibility spike in docs/design/matlab-http-transport.md (not part of the toolbox).
% Drives the handshake of a REAL local Studio service with matlab.net.http only: no Python inside MATLAB, no pyenv.
% Needs REPO_ROOT and SPIKE_WORKSPACE (a workspace whose service run_spike.sh started).
import matlab.net.http.*
import matlab.net.http.field.*
addpath(fullfile(getenv('REPO_ROOT'), 'Matlab', 'toolbox'));
workspace = getenv('SPIKE_WORKSPACE');
standinPort = str2double(getenv('SPIKE_STANDIN_PORT'));
rss = @() readStatus('VmRSS');
fprintf('SPIKE matlab %s, java heap max %d MB\n', version, round(java.lang.Runtime.getRuntime().maxMemory() / 2^20));

%% ---------------------------------------------------------------- Part A: the real service handshake
lock = jsondecode(fileread(fullfile(workspace, '.studio.lock')));
base = sprintf('http://127.0.0.1:%d', lock.port);
warning('off', 'MATLAB:http:BodyExpectedFor');          % an empty POST body (the mint request) is legitimate
opts = HTTPOptions('UseProxy', false, 'ConnectTimeout', 10, 'ResponseTimeout', 30);

req = RequestMessage('POST', HeaderField('X-OpenDPD-Launcher', lock.launcher_secret), MessageBody(uint8([])));
tic; resp = send(req, [base '/bootstrap/mint'], opts); tMint = toc;
fprintf('SPIKE A1 mint: status %d, content-type %s, %.0f ms\n', double(resp.StatusCode), char(string(resp.Header.getFields('Content-Type').Value)), 1000 * tMint);
token = resp.Body.Data.token;

body = MessageBody(struct('token', token));
req = RequestMessage('POST', [ContentTypeField('application/json'), HeaderField('Accept', 'application/json')], body);
[resp, completed] = send(req, [base '/api/v1/session/bootstrap'], opts);
csrf = resp.Body.Data.csrf_token;
setCookies = resp.getFields('Set-Cookie');
fprintf('SPIKE A2 session bootstrap: status %d, Set-Cookie fields %d, csrf token length %d\n', double(resp.StatusCode), numel(setCookies), strlength(string(csrf)));
cookieInfo = setCookies.convert();
fprintf('SPIKE A2 cookie name %s, value length %d, path %s, extensions %s\n', string(cookieInfo(1).Cookie.Name), strlength(string(cookieInfo(1).Cookie.Value)), string(cookieInfo(1).Path), strjoin(cookieInfo(1).Extensions, ','));
cookie = CookieField([cookieInfo.Cookie]);

req = RequestMessage('GET', [HeaderField('Accept', 'application/json'), HeaderField('X-OpenDPD-CSRF', csrf), cookie]);
[resp, completed] = send(req, [base '/api/v1/system/capabilities'], opts);
fprintf('SPIKE A3 capabilities with the cookie by hand: status %d, version %s, workspace matches %d\n', double(resp.StatusCode), resp.Body.Data.version, strcmp(resp.Body.Data.workspace, workspace));

req = RequestMessage('GET', [HeaderField('Accept', 'application/json'), HeaderField('X-OpenDPD-CSRF', csrf)]);
[resp, completed] = send(req, [base '/api/v1/system/capabilities'], opts);
fprintf('SPIKE A4 capabilities WITHOUT the cookie: status %d\n', double(resp.StatusCode));

[resp, completed] = send(RequestMessage('GET', [HeaderField('Accept', 'application/json'), cookie]), [base '/api/v1/system/capabilities'], opts);
fprintf('SPIKE A5 GET with the cookie and no CSRF header: status %d (GET needs no CSRF)\n', double(resp.StatusCode));

req = RequestMessage('POST', [ContentTypeField('application/json'), HeaderField('Accept', 'application/json'), cookie], MessageBody(struct('x', 1)));
[resp, completed] = send(req, [base '/api/v1/datasets/import'], opts);
code = '';
try, code = resp.Body.Data.error.code; catch, end
fprintf('SPIKE A6 mutating POST with the cookie and WITHOUT the CSRF header: status %d, error code %s\n', double(resp.StatusCode), string(code));

req = RequestMessage('POST', [ContentTypeField('application/json'), HeaderField('Accept', 'application/json'), HeaderField('X-OpenDPD-CSRF', csrf), cookie], MessageBody(struct('x', 1)));
[resp, completed] = send(req, [base '/api/v1/datasets/import'], opts);
code = '';
try, code = resp.Body.Data.error.code; catch, end
fprintf('SPIKE A7 mutating POST with cookie and CSRF but an invalid body: status %d, error code %s (validation, so the boundary accepted it)\n', double(resp.StatusCode), string(code));

hostChecked = RequestMessage('GET', [HeaderField('Accept', 'application/json'), HeaderField('Host', 'evil.example:80'), cookie]);
try
    [resp, completed] = send(hostChecked, [base '/api/v1/system/capabilities'], opts);
    fprintf('SPIKE A8 forged Host header: status %d\n', double(resp.StatusCode));
catch err
    fprintf('SPIKE A8 forged Host header could not be sent: %s\n', err.identifier);
end


fprintf('SPIKE done\n');
