## Specify what command to use to invoke a web browser when opening the notebook.
#  If not specified, the default browser will be determined by the `webbrowser`
#  standard library module, which allows setting of the BROWSER environment
#  variable to override it.
# c.NotebookApp.browser = 'chromium-browser --incognito %s'
c.NotebookApp.browser = '/usr/bin/google-chrome-stable --incognito %s'

## The IP address the notebook server will listen on.
c.NotebookApp.ip = '0.0.0.0' # listen on all IPs

## The port the notebook server will listen on.
c.NotebookApp.port = 9999

## Token used for authenticating first-time connections to the server.
#  When no password is enabled, the default is to generate a new, random token.
#  Setting to an empty string disables authentication altogether, which is NOT
#  RECOMMENDED.
c.NotebookApp.token = ''     # disable authentication

## Set the Access-Control-Allow-Origin header
#  Use '*' to allow any origin to access your server.
#  Takes precedence over allow_origin_pat.
c.NotebookApp.allow_origin = '*' # allow access from anywhere

## Hashed password to use for web authentication.
#  To generate, type in a python/IPython shell:
#    from notebook.auth import passwd; passwd()
#  The string should be of the form type:salt:hashed-password.
c.NotebookApp.password = 'sha1:6cbfb750f755:7ba239ff64cdf6671b12fab06886ceb4baacc65f'


# ## The directory for user settings.
# c.LabApp.user_settings_dir = cur_path + '/.jupyter/lab/user-settings'

# ## The directory for workspaces
# c.LabApp.workspaces_dir = cur_path + '/.jupyter/lab/workspaces'

import os
cur_path = os.path.dirname(os.path.abspath(__file__))

## The directory to use for notebooks and kernels.
c.NotebookApp.notebook_dir = cur_path + "/../"

## The directory for user settings.
c.LabApp.user_settings_dir = cur_path + '/.jupyter/lab/user-settings'

## The directory for workspaces
c.LabApp.workspaces_dir = cur_path + '/.jupyter/lab/workspaces'