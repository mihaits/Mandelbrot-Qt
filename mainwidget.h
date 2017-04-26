#ifndef MAINWIDGET_H
#define MAINWIDGET_H

#include <QWidget>
#include <QLabel>

//declarații foward

class QVBoxLayout;
class QHBoxLayout;
class QImage;
class QPushButton;
class QProgressBar;

class MyLabel : public QLabel   //pentru a afișa o imagine în qt de obicei se folosește un label căruia i se asociază un pixmap. MyLabel e o subclasă a QLabel făcută pentru a putea obține coordonate relative cu labelul ale click-urilor pe el definind mousePressEvent
{
    Q_OBJECT
signals:

public:

private:
};

class MainWidget : public QWidget //fereastra principală
{
    Q_OBJECT
public:
    explicit MainWidget(QWidget *parent = 0);

    void BuildUi();
    void UpdatePixmap();

    //primește coordonatele unui pixel în fereastră, calculează în funcție de origine și zoom și returnează coordonatele absolute ale acestuia în fractal
    QPointF GraphCoord( const QPoint& );
    void BuildMandelbrot();
    QRgb Iterate( const double&, const double& );
    void UpdateInfoLabel();

    //funcția pentru zoom in. calculează noile coordonate în funcție de poziția click-ului astfel încât locul în care s-a dat click rămâne în aceeași poziție
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
private:
    QVBoxLayout* vbLay;
    QHBoxLayout* hbLay;
    QImage* image;
    QPushButton* pushReset, * pushOptions;
    MyLabel* imgLabel;
    QLabel* infoLabel, * infoLabel2;
    QProgressBar* progBar;
    bool showProgbar = true, showInfo = true, isNormalized = false, saveImg = false;

    double zoomFactor = 1, xOrigin = 0, yOrigin = 0, zoomStep = 1.5, moveStep = 1;
    int maxIterations = 50;

};





#endif // MAINWIDGET_H
