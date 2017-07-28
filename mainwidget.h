#ifndef MAINWIDGET_H
#define MAINWIDGET_H

#include <QWidget>

class QLabel;
class QVBoxLayout;
class QHBoxLayout;
class QImage;
class QPushButton;
class QProgressBar;

class MainWidget : public QWidget
{
    Q_OBJECT

public:
    explicit MainWidget(QWidget *parent = 0);

    void BuildUi();
    void UpdatePixmap();

    QPointF pixelCoordsToGraphCoords( const QPoint& );
    void BuildMandelbrot();
    QRgb Iterate( const double&, const double& );
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
    void ToggleProgBar();
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
    QProgressBar* progBar;
    bool showProgbar = true, showInfo = true, isNormalized = false, saveImg = false;

    double zoomFactor = 1, xOrigin = 0, yOrigin = 0, zoomStep = 1.5, moveStep = 1;
    int maxIterations = 50;

};





#endif //MAINWIDGET_H
