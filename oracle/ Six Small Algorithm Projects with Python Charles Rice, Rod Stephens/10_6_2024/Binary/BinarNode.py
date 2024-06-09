import tkinter as tk
class BinarNode:
    indent = "  "
    node_radius=10
    x_spacing = 20
    y_spacing = 20
    def __init__(self, value):
        self.value = value
        self.left_child = None
        self.right_child = None
        self.center =(0,0)
        self.subtree_bounds=(
            self.center[0]-BinarNode.node_radius,
            self.center[1]-BinarNode.node_radius,
            self.center[0]+BinarNode.node_radius,
            self.center[1]+BinarNode.node_radius
        )

    def add_left(self, child):
        self.left_child = child

    def add_right(self, child):
        self.right_child = child
    def __str__(self,level=0):
        result = level * BinarNode.indent + f"{self.value}:\n"
        if (self.left_child is not None) or (self.right_child is not None):
            if self.left_child is None:
                result += f"{(level+1)*BinarNode.indent}None\n"
            else:
                result+=self.left_child.__str__(level+1)
            if self.right_child is None:
                result += f"{(level+1)*BinarNode.indent}None\n"
            else:
                result+=self.right_child.__str__(level+1)
        return result
    def find_node(self,target):
        if self.value==target:
            return self
        if self.left_child is not None:
            result=self.left_child.find_node(target)
            if result is not None:
                return result
        if self.right_child is not None:
            result=self.right_child.find_node(target)
            if result is not None:
                return result
        return None
    def traverse_preorder(self):
        result=[self]
        if self.left_child is not None:
            result+=self.left_child.traverse_preorder()
        if self.right_child is not None:
            result+=self.right_child.traverse_preorder()
        return result
    def traverse_postorder(self):
        result=[]
        if self.left_child is not None:
            result+=self.left_child.traverse_postorder()
        if self.right_child is not None:
            result+=self.right_child.traverse_postorder()
        result.append(self)
        return result
    def traverse_inorder(self):
        result=[]
        if self.left_child is not None:
            result+=self.left_child.traverse_inorder()
        result.append(self)
        if self.right_child is not None:
            result+=self.right_child.traverse_inorder()
        return result
    def traverse_breadth_first(self):
        resut=[]
        queue=[self]
        while len(queue)>0:
            node=queue.pop(0)
            resut.append(node)
            if node.left_child is not None:
                queue.append(node.left_child)
            if node.right_child is not None:
                queue.append(node.right_child)
        return resut
    def arrange_subtree(self,xmin,ymin):
        cy=ymin +BinarNode.node_radius
        
        if (self.left_child is None) and (self.right_child is None):
            cx=xmin+BinarNode.node_radius
            self.center=(cx,cy)
            xmax=xmin + 2*BinarNode.node_radius
            ymax=ymin + 2*BinarNode.node_radius
            self.subtree_bounds=(xmin,ymin,xmax,ymax)
            return
        child_xmin=xmin
        child_ymin=ymin+ 2*BinarNode.node_radius +BinarNode.y_spacing
        if self.left_child is not None:
            self.left_child.arrange_subtree(child_xmin,child_ymin)
            child_xmin=self.left_child.subtree_bounds[2]
            if self.right_child is not None:
                child_xmin+=BinarNode.x_spacing
        if self.right_child is not None:
            self.right_child.arrange_subtree(child_xmin, child_ymin)
        if (self.left_child is not None) and (self.right_child is not None):
            cx=(self.left_child.center[0] + self.right_child.center[0])/2
            self.center= (cx,cy)
            xmax=self.right_child.subtree_bounds[2]
            ymax=max(self.right_child.subtree_bounds[3],self.left_child.subtree_bounds[3])
            self.subtree_bounds=(xmin,ymin,xmax,ymax)
        elif self.left_child is not None:
            cx=self.left_child.center[0]
            self.center=(cx,cy)
            xmax=self.left_child.subtree_bounds[2]
            ymax=self.left_child.subtree_bounds[3]
            self.subtree_bounds=(xmin,ymin,xmax,ymax)
        else:
            cx=self.right_child.center[0]
            self.center=(cx,cy)
            xmax=self.right_child.subtree_bounds[2]
            ymax=self.right_child.subtree_bounds[3]
            self.subtree_bounds=(xmin,ymin,xmax,ymax)
            
    def draw_subtree_links(self,canvas):
        if self.left_child is not None:
            self.left_child.draw_subtree_links(canvas)
            canvas.create_line(
                self.center[0],self.center[1],
                self.left_child.center[0],self.left_child.center[1],
                fill="black")
        if self.right_child is not None:
            self.right_child.draw_subtree_links(canvas)
            canvas.create_line(self.center[0],self.center[1],
            self.right_child.center[0],self.right_child.center[1],
            fill="black")
    def draw_subtree_node(self,canvas):
        x0=self.center[0]-BinarNode.node_radius
        y0=self.center[1]-BinarNode.node_radius
        x1=self.center[0]+BinarNode.node_radius
        y1=self.center[1]+BinarNode.node_radius
        
        canvas.create_oval(x0,y0,x1,y1,fill="white",outline='green')
        canvas.create_text(self.center,text=self.value,fill="red")
        if self.left_child is not None:
            self.left_child.draw_subtree_node(canvas)
        if self.right_child is not None:
            self.right_child.draw_subtree_node(canvas)
    def arrange_and_draw_subtree(self,canvas,xmin,ymin):
        self.arrange_subtree(xmin,ymin)
        self.draw_subtree_links(canvas)
        self.draw_subtree_node(canvas)
def find_value(root,target):
    node=root.find_node(target)
    if node is None:
        print(f"value {target} not found")
    else:
        print(f"value {target} found")     

def kill_callback():
    window.destroy()
#==========================================
#outside the class code    
root= BinarNode("Root")
a= BinarNode("a")
b= BinarNode("b")
c= BinarNode("c")
d= BinarNode("d")
e= BinarNode("e")
f= BinarNode("f")
root.add_left(a)
root.add_right(b)
a.add_left(c)
a.add_right(d)
b.add_right(e)
e.add_left(f)
# print(root)
# print(a)
# find_value(root,"a")
# print("Preorder:  ",end='')
# for node in root.traverse_preorder():
#     print(f'{node.value} ',end='')
# print()

# print("Postorder: ",end='')
# for node in root.traverse_postorder():
#     print(f'{node.value} ',end='')
# print()

# print("Inorder:   ",end='')
# for node in root.traverse_inorder():
#     print(f'{node.value} ',end='')
# print()

# print("Breadth First: ",end='')
# for node in root.traverse_breadth_first():
#     print(f'{node.value} ',end='')
# print()

window=tk.Tk()
window.title('Binary_nodes')
window.protocol("WM_DELETE_WINDOW",kill_callback)
window.geometry("260x220")
canvas=tk.Canvas(window, bg='white',borderwidth=2,relief=tk.SUNKEN)
canvas.pack(padx=10,pady=10,fill=tk.BOTH,expand=True)
root.arrange_and_draw_subtree(canvas,10,10)
window.focus_force()
window.mainloop()