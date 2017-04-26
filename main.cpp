/*
      MANDELBROTQT, un program care desenează fractalul Mandelbrot în Qt

Funcționalități:
 * controlarea numărului de iterații prin tastele plus și minus
 * zoom in în locul în care s-a dat click stânga (în limita impusă de precizia variabilelor double, ~10^14)
 * zoom out prin click dreapta
 * deplasarea poziției cu tastele săgeți
 * ajustaje fine la ținerea tastei shift
 * resetarea poziției și a numărului de iterații
 * funcții comutabile:
 *      salvare imagine
 *      informații iterație, zoom poziție
 *      bară de progres în construirea fractalului
 *      normalizarea imaginii

Logică de funcționare (se presupune înțelegerea mulțimii/fractalului Mandelbrot):
 * în orice moment fereastra programului afișează o anumită zonă din fractalul Mandelbrot. Aceasta zonă poate fi definită prin următoarele informații (reprezentate de variabile locale private în clasa ferestrei principale, MainWidget):
    - coordonatele în spațiul numerelor complexe a centrului zonei afișate (xOriginX, yOrigin)
    - nivelul de zoom (zoomFactor, când este egal cu 1 înseamnă că fereastra cuprinde zona dintre coordonatele (-2,-2) și (2,2) )
    - numărul maxim de iterații efectuate în construirea fractalului
 * când este construit fractalul, fiecare pixel din fereastra afișată este iterat individual: coordonatele fiecaruia sunt convertite la coordonatele absolute corespuzătoare în spațiul numerelor complexe pentru a fi folosite în calcularea iterațiilor, în funcție de datele de mai sus. pixelii sunt colorați cu un gradient de la alb la negru în funcție de numărul de iterații care le corespunde după calcularea formulei.
 * fractalul este reconstruit și afișat când are loc un eveniment care are efect asupra poziției sau aspectului fractaluli din fereastră, modificând variabilele enumerate. procesul este:
    - eveniment -> modificare corespunzătoare a variabilelor care definesc zona afișată -> reiterarea pixelilor -> actualizarea imaginii
*/

/*TODO:
 * modularizare
 * optimizări
 * salvare poziție
 * culori mai complexe: gradiente de mai multe culori, gradiente între iterații
 * redimensionare fereastră
 */
#include <QApplication>
#include "mainwidget.h"

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    MainWidget widget;

    widget.show();

    return a.exec();
}
