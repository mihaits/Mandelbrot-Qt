#ifndef MAINWIDGET_H
#define MAINWIDGET_H

#include <QWidget>

class QLabel;
class QVBoxLayout;
class QHBoxLayout;
class QImage;
class QPushButton;

class MainWidget : public QWidget
{
    Q_OBJECT

public:
    explicit MainWidget(QWidget *parent = 0);

    void BuildUi();
    void UpdatePixmap();

    void BuildMandelbrot();
    void UpdateInfoLabel();

    void ZoomIn( const QPoint& );
    void ZoomOut();
    void Normalize();

signals:

public slots:
    void Reset();
    void OpenOptions();
    void ToggleInfo();
    void ToggleSave();
    void ToggleNormalization();

protected:
    bool eventFilter(QObject*, QEvent*);
    void keyPressEvent(QKeyEvent*);

private:
    QVBoxLayout* mainVbLay;
    QHBoxLayout* buttonsHbLay;
    QImage* image;
    QPushButton* pushReset, * pushOptions;
    QLabel* imgLabel;
    QLabel* zoomInfoLabel, * posInfoLabel;
    bool isNormalized = false;
    bool saveImg = false;
    bool showInfo = true;

    double zoomFactor = 1;
    double xOrigin = 0;
    double yOrigin = 0;
    double zoomStep = 1.5;
    double moveStep = 1;
    int maxIterations = 50;

};





#endif //MAINWIDGET_H
