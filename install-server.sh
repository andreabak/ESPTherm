#!/bin/bash

set -e

installdir_warning() {
  echo "Installed server service will run from current directory: ${INSTALLDIR}"
  echo "This directory must be permanent on the filesystem, or the service will fail to start."
  echo "If you want to use another install directory, please copy all the files there and re-run this script."
  while true; do
    read -r -p "Proceed, confirming current directory? [yes/no]: " yn
    case $yn in
    [Yy]es) break ;;
    [Nn]*) exit ;;
    *) echo "Please answer yes or no." ;;
    esac
  done
}

set_prompt_optional() {
  varname=$1
  prompttext=$2

  read -r -p "${prompttext} (current: \"${!varname}\", blank to confirm): " inputvalue
  if [ -n "${inputvalue}" ]; then
    declare "${varname}=${inputvalue}"
    echo "${varname} set to \"${!varname}\""
  fi
}

create_runenv() {
  echo "Setting up python runtime venv"
  if [ ! -d "${RUNENVDIR}" ]; then
    (
      python3 -m venv "${RUNENVDIR}"
      "${RUNENVDIR}/bin/python" -m pip install -r server_requirements.txt
    )
    if [ ! $? ]; then
      echo "Error while setting up runtime venv, rolling back"
      rm -rf "${RUNENVDIR}"
    fi
  fi
}

INSTALLDIR="$(pwd)"
installdir_warning
RUNENVDIR="${INSTALLDIR}/.runenv"
RUNUSER=$(whoami)
set_prompt_optional RUNUSER "[Optional] specify user to run server daemon"
SERVERADDR="0.0.0.0"
set_prompt_optional SERVERADDR "[Optional] specify address the server will bind to"
SERVERPORT=8367
set_prompt_optional SERVERPORT "[Optional] specify port the server will listen on"

create_runenv

SERVICEFINALFILENAME="esptherm-server.service"
SERVICETEMPLATEFILE="esptherm-server.service.template"
SERVICETEMPFILE="${SERVICETEMPLATEFILE}.TEMP"
set +e
(
  set -e
  echo "Compiling systemd service"
  (
    export INSTALLDIR RUNENVDIR RUNUSER SERVERADDR SERVERPORT
    envsubst <"${SERVICETEMPLATEFILE}" >"${SERVICETEMPFILE}"
  )
  read -r -p "Do you want to open the systemd service file for editing before install? [yes/no (default)]: " yn
  if [[ $yn == [Yy]* ]]; then
    DEFAULTEDITOR="$(command -v nano >/dev/null 2>&1 && echo 'nano' || echo 'vi')"
    "${FCEDIT:-${VISUAL:-${EDITOR:-$DEFAULTEDITOR}}}" "${SERVICETEMPFILE}"
  fi
  echo "Installing and starting systemd service"
  sudo cp -i "${SERVICETEMPFILE}" -t "/etc/systemd/system/${SERVICEFINALFILENAME}"
)
result=$?
set -e
rm "${SERVICETEMPFILE}" # finally
((!result))             # eval return code
sudo systemctl daemon-reload
sudo systemctl enable esptherm-server
sudo systemctl restart esptherm-server
echo "Done."
