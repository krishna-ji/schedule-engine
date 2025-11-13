
Tmux Basics:
New session	: tmux new -s <name>
List sessions: tmux ls
Attach session: tmux attach -t <name>
Detach session: Ctrl + b, then d
Kill session: tmux kill-session -t <name>

tmux Windows
New window	Ctrl + b, then c
Next window	Ctrl + b, then n
Previous window	Ctrl + b, then p
Rename window	Ctrl + b, then ,
Close window	exit (inside window) or Ctrl + b, then &

Panes (Split screen)
Split vertically	Ctrl + b, then %
Split horizontally	Ctrl + b, then "
Switch pane			Ctrl + b, then arrow keys
Resize pane			Ctrl + b, then hold arrow keys
Close pane	exit (inside pane)

Enter scroll mode	Ctrl + b, then [
Move up/down	Arrow keys or PgUp / PgDn
Scroll faster	Ctrl + u (up half page), Ctrl + d (down half page)
Exit scroll mode	Press q
Copy text (optional)	Press Space to start selecting → move cursor → press Enter to copy
Paste copied text	Ctrl + b, then ]