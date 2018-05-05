
# added by Anaconda3 5.1.0 installer
export PATH="/Users/david/anaconda3/bin:$PATH"
export PATH="~/bin:$PATH"
#export PATH="$PATH:$GOPATH"

##

alias ls='ls -F'

#export PS1="[david:\w] "

## Git Changes
git config --global user.email "david.g.whiting@gmail.com"
git config --global user.name "David Whiting"

# get current branch in git repo
function parse_git_branch() {
	BRANCH=`git branch 2> /dev/null | sed -e '/^[^*]/d' -e 's/* \(.*\)/\1/'`
	if [ ! "${BRANCH}" == "" ]
	then
		STAT=`parse_git_dirty`
		echo "(${BRANCH}${STAT})"
	else
		echo ""
	fi
}

# Get current status of git repo
function parse_git_dirty {
	status=`git status 2>&1 | tee`
	dirty=`echo -n "${status}" 2> /dev/null | grep "modified:" &> /dev/null; echo "$?"`
	untracked=`echo -n "${status}" 2> /dev/null | grep "Untracked files" &> /dev/null; echo "$?"`
	ahead=`echo -n "${status}" 2> /dev/null | grep "Your branch is ahead of" &> /dev/null; echo "$?"`
	newfile=`echo -n "${status}" 2> /dev/null | grep "new file:" &> /dev/null; echo "$?"`
	renamed=`echo -n "${status}" 2> /dev/null | grep "renamed:" &> /dev/null; echo "$?"`
	deleted=`echo -n "${status}" 2> /dev/null | grep "deleted:" &> /dev/null; echo "$?"`
	bits=''
	if [ "${renamed}" == "0" ]; then
		bits=">${bits}"
	fi
	if [ "${ahead}" == "0" ]; then
		bits="*${bits}"
	fi
	if [ "${newfile}" == "0" ]; then
		bits="+${bits}"
	fi
	if [ "${untracked}" == "0" ]; then
		bits="?${bits}"
	fi
	if [ "${deleted}" == "0" ]; then
		bits="x${bits}"
	fi
	if [ "${dirty}" == "0" ]; then
		bits="!${bits}"
	fi
	if [ ! "${bits}" == "" ]; then
		echo " ${bits}"
	else
		echo ""
	fi
}

# include github branch and status
#export PS1="[david:\w]\`parse_git_branch\` "

# with red letters for github status
export PS1="[david:\w \[\e[31m\]\`parse_git_branch\`\[\e[m\]] "

# Add all of the following to your .bash_profile
 

# single sign on password as a session environment variable
#export SINGLE_SIGN_ON_PASSWORD=$(~/bin/get_sso_password_from_osx_keychain.sh)
