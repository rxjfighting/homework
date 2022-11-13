package test;

import java.io.IOException;

abstract class Geometric {//设计一个名为Geometric的几何图形的抽象类
    private String color;
    private boolean filled;//两个名为color、filled属性分别表示图形颜色和是否填充

    public Geometric() {
        ;
    }

    public Geometric(String color, boolean filled) {
        this.color = color;
        this.filled = filled;
    }

    public String getColor() {
        return color;
    }

    public void setColor(String color) {
        this.color = color;
    }

    public boolean getFilled() {
        return filled;
    }

    public void setFilled(boolean filled) {
        this.filled = filled;
    }


    abstract double getArea();//一个名为getArea()的抽象方法，返回图形的面积。

    abstract double getPerimeter();//一个名为getPerimeter()的抽象方法，返回图形的周长。


}
class Circle extends Geometric {
    private double radius;

    public Circle() {
        ;
    }

    public Circle(String color, boolean filled, double radius) {
        super(color, filled);//调用构造方法
        this.radius = radius;
    }

    public double getRadius() {
        return radius;
    }

    public void setRadius(double radius) {
        this.radius = radius;
    }

    public double getArea() {
        return Math.PI * radius * radius;
    }

    public double getPerimeter() {
        return 2 * Math.PI * radius;
    }

    public String toString() {
        return "半径为" + radius + "，面积为" + getArea() + "，周长为" + getPerimeter();
    }
}
class Rectangle extends Geometric {


    private double side1;
    private double side2;

    public Rectangle() {
        ;
    }

    public Rectangle(String color, boolean filled, double side1, double side2) {
        super(color, filled);
        this.side1 = side1;
        this.side2 = side2;
    }

    public double getSide1() {
        return side1;
    }

    public void setSide1(double side1) {
        this.side1 = side1;
    }

    public double getSide2() {
        return side2;
    }

    public void setSide2(double side2) {
        this.side2 = side2;
    }

    public double getArea() {
        return side1 * side2;
    }

    public double getPerimeter() {
        return 2 * side1 + 2 * side2;
    }
    public String toString() {
        return "长为" + side1 + "，宽为" + side2 + "，面积为" + getArea() + "，周长为" + getPerimeter();
    }
}
class Triangle extends Geometric {
    private double side1;
    private double side2;
    private double side3;

    public Triangle() {
        ;
    }

    public Triangle(String color, boolean filled, double side1, double side2, double side3) {
        super(color, filled);
        this.side1 = side1;
        this.side2 = side2;
        this.side3 = side3;
    }

    public double getSide1() {
        return side1;
    }

    public void setSide1(double side1) {
        this.side1 = side1;
    }

    public double getSide2() {
        return side2;
    }

    public void setSide2(double side2) {
        this.side2 = side2;
    }

    public double getSide3() {
        return side3;
    }

    public void setSide3(double side3) {
        this.side3 = side3;
    }

    public double getArea() {
        double s = side1 + side2 + side3 / 2;
        float S = (float) Math.sqrt(s * (s - side1) * (s - side2) * (s - side3));
        return S;
    }

    public double getPerimeter() {
        return side1 + side2 + side3;
    }

    public boolean isTriangle(double side1, double side2, double side3) {
        if (side1 + side2 > side3 && side1 + side3 > side2 && side2 + side3 > side1) {
            return true;
        } else {
            return false;
        }
    }

    public String toString() {
        double min;
        if (side1 <= side2 && side1 <= side3) {
            min = side1;
        } else if (side2 <= side1 && side2 <= side3) {
            min = side2;
        } else {
            min = side3;
        }
        return min + "为最小边";
    }
}
public class Test {
    public static void main(String[] args) {
        Geometric g1 = new Circle("red", true, 3.5);
        Geometric g2 = new Rectangle("blue", true, 1.5, 2.0);
        Triangle g3 = new Triangle("yellow", false, 2, 2, 2);
        System.out.println("圆"+g1.toString());
        System.out.println("矩形"+g2.toString());
        System.out.print("三角形面积为"+g3.getArea()+"，周长为"+g3.getPerimeter()+"，此外"+g3.toString());
    }
}
