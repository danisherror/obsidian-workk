import tkinter as tk
class NaryNode:
    indent = "  "
    node_radius=10
    x_spacing = 20
    y_spacing = 20
    def __init__(self,value):
        self.value= value
        self.children= []
        self.center =(0,0)
        self.subtree_bounds=(
            self.center[0]-NaryNode.node_radius,
            self.center[1]-NaryNode.node_radius,
            self.center[0]+NaryNode.node_radius,
            self.center[1]+NaryNode.node_radius
        )
    def add_child(self,child):
        self.children.append(child)
    def __str__(self, level=0):
        result= level* NaryNode.indent + f"{self.value}:\n"
        for child in self.children:
            result+= child.__str__(level+1)
        return result
    def find_node(self,target):
        if self.value == target:
            return self
        for child in self.children:
            result= child.find_node(target)
            if result is not None:
                return result
        return None
    def traverse_preorder(self):
        result=[self]
        for child in self.children:
            result+=child.traverse_preorder()
        return result
    def traverse_postorder(self):
        result=[]
        for child in self.children:
            result+=child.traverse_postorder()
        result.append(self)
        return result
    def traverse_breadth_first(self):
        result=[]
        queue=[self]
        while len(queue) >0:
            node=queue.pop(0)
            result.append(node)
            for child in node.children:
                queue.append(child)
        return result
    def arrange_subtree(self,xmin,ymin):
        cy=ymin+NaryNode.node_radius
        if len(self.children)==0:
            cx=xmin+NaryNode.node_radius
            self.center=(cx,cy)
            xmax=xmin+ 2*NaryNode.node_radius
            ymax=ymin+2*NaryNode.node_radius
            self.subtree_bounds=(xmin,ymin,xmax,ymax)
            return
        child_xmin=xmin
        child_ymin=ymin+2*NaryNode.node_radius +NaryNode.y_spacing
        ymax=ymin+2*NaryNode.node_radius
        for child in self.children:
            child.arrange_subtree(child_xmin,child_ymin)
            child_xmin=child.subtree_bounds[2]+NaryNode.x_spacing
            if ymax< child.subtree_bounds[3]:
                ymax=child.subtree_bounds[3]
        xmax=child_xmin- NaryNode.x_spacing
        self.subtree_bounds=(xmin,ymin,xmax,ymax)
        cx=(self.subtree_bounds[0]+self.subtree_bounds[2])/2
        cy=ymin+NaryNode.node_radius
        self.center=(cx,cy)
    def draw_subtree_links(self,canvas):
        if len(self.children)==1:
            child=self.children[0]
            canvas.create_line(self.center[0],self.center[1],
            child.center[0],child.center[1],fill="green")
        elif len(self.children) >0:
            ymid= (self.center[1]+self.children[0].center[1])/2
            canvas.create_line(self.center[0],self.center[1],
            self.center[0],ymid,fill="green")
            last_child=len(self.children) - 1
            canvas.create_line(self.children[0].center[0],ymid,
            self.children[last_child].center[0],ymid,fill="green")
            for child in self.children:
                canvas.create_line(child.center[0],ymid,
                child.center[0],child.center[1],fill="green")
                
        for child in self.children:
            child.draw_subtree_links(canvas)
    def draw_subtree_node(self,canvas):
        x0=self.center[0]-NaryNode.node_radius
        y0=self.center[1]-NaryNode.node_radius
        x1=self.center[0]+NaryNode.node_radius
        y1=self.center[1]+NaryNode.node_radius
        
        canvas.create_oval(x0,y0,x1,y1,fill="white",outline='black')
        canvas.create_text(self.center,text=self.value,fill="red")
        for child in self.children:
            child.draw_subtree_node(canvas)
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
root= NaryNode("Root")
a= NaryNode("a")
b= NaryNode("b")
c= NaryNode("c")
d= NaryNode("d")
e = NaryNode("e")
f= NaryNode("f")
g= NaryNode("g")
h= NaryNode("h")
i= NaryNode("i")

root.add_child(a)
root.add_child(b)
root.add_child(c)

a.add_child(d)
a.add_child(e)
c.add_child(f)
d.add_child(g)
f.add_child(h)
f.add_child(i)


# print(root)
# find_value(root,"a")
# print("Preorder:  ",end='')
# for node in root.traverse_preorder():
#     print(f'{node.value} ',end='')
# print()

# print("Postorder: ",end='')
# for node in root.traverse_postorder():
#     print(f'{node.value} ',end='')
# print()

# print("Breadth First: ",end='')
# for node in root.traverse_breadth_first():
#     print(f'{node.value} ',end='')
# print()


window=tk.Tk()
window.title('Binary_nodes')
window.protocol("WM_DELETE_WINDOW",kill_callback)
window.geometry("260x180")
canvas=tk.Canvas(window, bg='white',borderwidth=2,relief=tk.SUNKEN)
canvas.pack(padx=10,pady=10,fill=tk.BOTH,expand=True)
root.arrange_and_draw_subtree(canvas,10,10)
window.focus_force()
window.mainloop()
