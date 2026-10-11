"""deck_style.py -- design system for the CISO presentation.

One place for the palette, fonts, and reusable slide primitives so every slide
reads as one system. The template follows the McKinsey consulting idiom, tuned
to be white-dominant: white content slides, near-black charcoal text, charcoal
title and closing slides, and a single green accent used sparingly (rules, flow
steps, plotted points). Georgia display titles with Arial body, informative
action titles, thin hairline rules, and source-plus-page-number footers.
"""

from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import MSO_AUTO_SIZE
from pptx.oxml.ns import qn

# ---- Canvas (16:9) ----
SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)

# ---- Palette: white-dominant, charcoal primary, green as a restrained accent ----
# Token names are historical; the ROLE each plays is what matters now. Mostly
# white with near-black charcoal text and one green accent used sparingly.
PLUM      = RGBColor(0x1F, 0x29, 0x33)   # charcoal slate: dark slides / titles / big figures
PLUM_DK   = RGBColor(0x14, 0x1B, 0x21)   # deeper charcoal for layering
TERRA     = RGBColor(0x2E, 0x7D, 0x64)   # green: the single restrained accent
TERRA_DK  = RGBColor(0x24, 0x61, 0x4E)
PAPER     = RGBColor(0xFF, 0xFF, 0xFF)   # white content background
CARD      = RGBColor(0xFF, 0xFF, 0xFF)   # white card
CARD_TINT = RGBColor(0xF4, 0xF6, 0xF5)   # faint cool-gray card fill
INK       = RGBColor(0x1A, 0x1A, 0x1A)   # near-black body text
INK_SOFT  = RGBColor(0x6B, 0x72, 0x80)   # gray captions / secondary
SAGE      = RGBColor(0x7C, 0x93, 0x8B)   # muted gray-green (low risk)
GOLD      = RGBColor(0xC8, 0x9B, 0x3C)   # (retained for the risk-matrix scale)
LINE_SOFT = RGBColor(0xD9, 0xDC, 0xDE)   # hairline separators on white
ON_PLUM   = RGBColor(0xFF, 0xFF, 0xFF)   # white text on charcoal
ON_PLUM_SOFT = RGBColor(0xAF, 0xB8, 0xBF)  # muted light text on charcoal

# Risk heat scale (semantic green -> gold -> red; standard for a risk matrix)
RISK_HI  = RGBColor(0xA0, 0x2E, 0x22)
RISK_MED = GOLD
RISK_LO  = SAGE

# ---- Fonts (the McKinsey PowerPoint pairing: both ship with Office, render on any machine) ----
F_HEAD = "Georgia"    # serif for titles / section headers / big figures
F_BODY = "Arial"      # clean sans for body, labels, captions


def _set_bg(slide, color):
    """Solid background fill for a slide."""
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color


def _no_line(shape):
    shape.line.fill.background()


def _soft_shadow(shape, blur=Pt(9), dist=Pt(3), alpha=72):
    """A restrained drop shadow via raw XML (python-pptx has no shadow API)."""
    spPr = shape._element.spPr
    # remove any existing effectLst
    for tag in ("a:effectLst",):
        for el in spPr.findall(qn(tag)):
            spPr.remove(el)
    effectLst = spPr.makeelement(qn("a:effectLst"), {})
    outer = effectLst.makeelement(qn("a:outerShdw"), {
        "blurRad": str(int(blur)),
        "dist": str(int(dist)),
        "dir": "5400000",   # straight down
        "rotWithShape": "0",
    })
    clr = outer.makeelement(qn("a:srgbClr"), {"val": "2B2426"})
    a = clr.makeelement(qn("a:alpha"), {"val": str(alpha * 1000)})
    clr.append(a)
    outer.append(clr)
    effectLst.append(outer)
    spPr.append(effectLst)


def add_text(slide, l, t, w, h, runs, *, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP,
             space_after=Pt(4), line_spacing=1.06, wrap=True, hanging=None):
    """Add a text box. `runs` is a list of paragraphs; each paragraph is a list of
    (text, opts) run tuples. opts keys: size, color, bold, italic, font, spacing.
    `hanging` (an EMU length) applies an APA hanging indent to each paragraph."""
    tb = slide.shapes.add_textbox(l, t, w, h)
    tf = tb.text_frame
    tf.word_wrap = wrap
    tf.auto_size = MSO_AUTO_SIZE.NONE
    tf.vertical_anchor = anchor
    tf.margin_left = 0
    tf.margin_right = 0
    tf.margin_top = 0
    tf.margin_bottom = 0
    for i, para in enumerate(runs):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_before = Pt(0)
        p.space_after = space_after
        p.line_spacing = line_spacing
        if hanging is not None:
            pPr = p._p.get_or_add_pPr()
            pPr.set("marL", str(int(hanging)))
            pPr.set("indent", str(-int(hanging)))
        for text, o in para:
            r = p.add_run()
            r.text = text
            f = r.font
            f.size = Pt(o.get("size", 16))
            f.bold = o.get("bold", False)
            f.italic = o.get("italic", False)
            f.name = o.get("font", F_BODY)
            f.color.rgb = o.get("color", INK)
            if "spacing" in o:
                _char_spacing(r, o["spacing"])
    return tb


def _char_spacing(run, pts):
    """Set letter spacing (spc in 1/100 pt) on a run."""
    rPr = run._r.get_or_add_rPr()
    rPr.set("spc", str(int(pts * 100)))


def add_card(slide, l, t, w, h, *, fill=CARD, line=None, shadow=True, radius=0.06):
    shp = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, l, t, w, h)
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill
    if line is None:
        _no_line(shp)
    else:
        shp.line.color.rgb = line
        shp.line.width = Pt(0.75)
    shp.shadow.inherit = False
    if shadow:
        _soft_shadow(shp)
    # adjust corner radius
    try:
        shp.adjustments[0] = radius
    except Exception:
        pass
    return shp


def add_circle(slide, cx, cy, d, *, fill=TERRA, text=None, text_color=None,
               size=18, bold=True, font=None, line=None):
    """Filled circle centered at (cx, cy) with diameter d. Optional centered text."""
    l = Emu(int(cx - d / 2))
    t = Emu(int(cy - d / 2))
    shp = slide.shapes.add_shape(MSO_SHAPE.OVAL, l, t, Emu(int(d)), Emu(int(d)))
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill
    if line is None:
        _no_line(shp)
    else:
        shp.line.color.rgb = line
        shp.line.width = Pt(1.0)
    shp.shadow.inherit = False
    if text is not None:
        tf = shp.text_frame
        tf.word_wrap = False
        tf.margin_left = 0
        tf.margin_right = 0
        tf.margin_top = 0
        tf.margin_bottom = 0
        tf.vertical_anchor = MSO_ANCHOR.MIDDLE
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        r = p.add_run()
        r.text = text
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.name = font or F_BODY
        r.font.color.rgb = text_color or ON_PLUM
    return shp


def add_rect(slide, l, t, w, h, *, fill=None, line=None, line_w=Pt(1.0), radius=None):
    shape = MSO_SHAPE.ROUNDED_RECTANGLE if radius is not None else MSO_SHAPE.RECTANGLE
    shp = slide.shapes.add_shape(shape, l, t, w, h)
    if fill is None:
        shp.fill.background()
    else:
        shp.fill.solid()
        shp.fill.fore_color.rgb = fill
    if line is None:
        _no_line(shp)
    else:
        shp.line.color.rgb = line
        shp.line.width = line_w
    shp.shadow.inherit = False
    if radius is not None:
        try:
            shp.adjustments[0] = radius
        except Exception:
            pass
    return shp


def add_connector(slide, x1, y1, x2, y2, *, color=TERRA, width=Pt(2.0), dash=None, arrow=False):
    from pptx.enum.shapes import MSO_CONNECTOR
    cxn = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, Emu(int(x1)), Emu(int(y1)),
                                     Emu(int(x2)), Emu(int(y2)))
    cxn.line.color.rgb = color
    cxn.line.width = width
    cxn.shadow.inherit = False
    ln = cxn.line._get_or_add_ln()
    if dash:
        d = ln.makeelement(qn("a:prstDash"), {"val": dash})
        ln.append(d)
    if arrow:
        he = ln.makeelement(qn("a:tailEnd"), {"type": "triangle", "w": "med", "len": "med"})
        ln.append(he)
    return cxn


def set_notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


def word_count(text):
    return len(text.split())
