def add_letter(ax, letter, x=-0.3, y=1.05, title=None, fontsize=7,**kwargs):
    text = "(" + letter + ")"
    if title is not None:
        text += " " + r"$\textbf{" + title + "}$"
    
    out = ax.text(x, y, text, transform=ax.transAxes,
                  va='bottom', ha='left', clip_on=False, fontsize=fontsize,
                  **kwargs)
    out.set_in_layout(False)
    return out
    