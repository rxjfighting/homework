package test;
class IllegalTriangleException extends Exception{
    public IllegalTriangleException(String message)
    {
        super(message);
    }
}
class Triangle1 {
    private double side1;
    private double side2;
    private double side3;

    public Triangle1(double side1, double side2, double side3)
            throws IllegalTriangleException {
        if (side1 + side2 > side3 && side1 + side3 > side2
                && side2 + side3 > side1) {
            this.side1 = side1;
            this.side2 = side2;
            this.side3 = side3;

        } else {
            throw new IllegalTriangleException("无法构成三角形");
        }
    }
}
public class Homework {
    public static void main(String[] args) {
        try {
            System.out.println("尝试建立三条边为1，2，5的三角形");
            Triangle1 a = new Triangle1(1, 2, 5);
        } catch (IllegalTriangleException e) {
            System.out.println(e.getMessage());
        }
    }
}
