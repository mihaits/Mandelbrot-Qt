#ifndef OPTIONSWIDGET_H
#define OPTIONSWIDGET_H

#include "mainwidget.h"
#include <QWidget>

class QVBoxLayout;
class QLabel;
class QCheckBox;
class QFrame;

class OptionsWidget : public QWidget
{
    Q_OBJECT
public:
    OptionsWidget( MainWidget* parent = 0, bool progbar = false, bool info = false, bool norm = false, bool save = false );

    void BuildUi();
    void closeEvent( QCloseEvent* event );

signals:

public slots:

private:
    QVBoxLayout* vbLay;
    QLabel* instLabel;
    QCheckBox* normCbox, * progbarCBox, * infoCBox, * saveCbox;
    QFrame* separator;
    MainWidget* parentWindow;

};

#endif // OPTIONSWIDGET_H
