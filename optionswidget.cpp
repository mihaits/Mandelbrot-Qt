#include "optionswidget.h"
#include <QLabel>
#include <QCheckBox>
#include <QVBoxLayout>
#include <QFrame>

void OptionsWidget::BuildUi()
{
    instLabel   = new QLabel(this);
    progbarCBox = new QCheckBox(this);
    infoCBox    = new QCheckBox(this);
    normCbox    = new QCheckBox(this);
    saveCbox    = new QCheckBox(this);
    vbLay       = new QVBoxLayout(this);
    separator   = new QFrame(this);

    instLabel   -> setText("Use left click to zoom in on an area.\n"
                           "Use right click to zoom out.\n"
                           "Use arow keys to move view.\n"
                           "Use plus and minus keys to change the number of iterations.\n"
                           "Hold Shift for small increments or Control for large increments.");
    normCbox    -> setText("Normalize color");
    progbarCBox -> setText("Show build progress bar");
    infoCBox    -> setText("Show view info");
    saveCbox    -> setText("Automatically save image in application folder");
    separator   -> setFrameShape(QFrame::HLine);
    separator   -> setFrameShadow(QFrame::Sunken);

    vbLay -> addWidget(instLabel);
    vbLay -> addWidget(separator);
    vbLay -> addWidget(progbarCBox);
    vbLay -> addWidget(infoCBox);
    vbLay -> addWidget(normCbox);
    vbLay -> addWidget(saveCbox);
    vbLay -> setSizeConstraint(QLayout::SetFixedSize);
    setLayout(vbLay);

    setWindowFlags(windowFlags() ^ Qt::WindowMaximizeButtonHint ^ Qt::WindowStaysOnTopHint);
}

void OptionsWidget::closeEvent(QCloseEvent* event)
{
    parentWindow -> setEnabled(true);
}

OptionsWidget::OptionsWidget(MainWidget* parent, bool progBar, bool info, bool norm, bool save) : QWidget()
{
    parentWindow = parent;

    BuildUi();

    if(progBar)
        progbarCBox -> setCheckState(Qt::Checked);
    if(info)
        infoCBox    -> setCheckState(Qt::Checked);
    if(norm)
        normCbox    -> setCheckState(Qt::Checked);
    if(save)
        saveCbox    -> setCheckState(Qt::Checked);

    connect(infoCBox,       SIGNAL(clicked()), parent, SLOT(ToggleInfo()));
    connect(saveCbox,       SIGNAL(clicked()), parent, SLOT(ToggleSave()));
    connect(progbarCBox,    SIGNAL(clicked()), parent, SLOT(ToggleProgBar()));
    connect(normCbox,       SIGNAL(clicked()), parent, SLOT(ToggleNormalization()));
}
