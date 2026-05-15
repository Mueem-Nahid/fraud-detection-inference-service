import os
import pulumi
import pulumi_aws as aws

PREFIX = "modelserve"
SSH_PUBLIC_KEY = os.environ.get("SSH_PUBLIC_KEY", "")
COMMON_TAGS = {"Project": "modelserve"}

ami = aws.ec2.get_ami(
    most_recent=True,
    owners=["amazon"],
    filters=[
        aws.ec2.GetAmiFilterArgs(name="name", values=["amzn2-ami-hvm-*-x86_64-gp2"]),
        aws.ec2.GetAmiFilterArgs(name="virtualization-type", values=["hvm"]),
    ],
)

key_pair = None
if SSH_PUBLIC_KEY:
    key_pair = aws.ec2.KeyPair(
        f"{PREFIX}-key",
        public_key=SSH_PUBLIC_KEY,
        tags={**COMMON_TAGS, "Name": f"{PREFIX}-key"},
    )

vpc = aws.ec2.Vpc(
    f"{PREFIX}-vpc",
    cidr_block="10.0.0.0/16",
    tags={**COMMON_TAGS, "Name": f"{PREFIX}-vpc"},
)

igw = aws.ec2.InternetGateway(
    f"{PREFIX}-igw",
    vpc_id=vpc.id,
    tags={**COMMON_TAGS, "Name": f"{PREFIX}-igw"},
)

subnet = aws.ec2.Subnet(
    f"{PREFIX}-subnet",
    vpc_id=vpc.id,
    cidr_block="10.0.1.0/24",
    map_public_ip_on_launch=True,
    tags={**COMMON_TAGS, "Name": f"{PREFIX}-subnet"},
)

route_table = aws.ec2.RouteTable(
    f"{PREFIX}-rt",
    vpc_id=vpc.id,
    routes=[
        aws.ec2.RouteTableRouteArgs(
            cidr_block="0.0.0.0/0",
            gateway_id=igw.id,
        )
    ],
    tags={**COMMON_TAGS, "Name": f"{PREFIX}-rt"},
)

aws.ec2.RouteTableAssociation(
    f"{PREFIX}-rt-assoc",
    subnet_id=subnet.id,
    route_table_id=route_table.id,
)

sg = aws.ec2.SecurityGroup(
    f"{PREFIX}-sg",
    vpc_id=vpc.id,
    ingress=[
        aws.ec2.SecurityGroupIngressArgs(
            protocol="tcp",
            from_port=22,
            to_port=22,
            cidr_blocks=["0.0.0.0/0"],
            description="SSH",
        ),
        aws.ec2.SecurityGroupIngressArgs(
            protocol="tcp",
            from_port=8000,
            to_port=8000,
            cidr_blocks=["0.0.0.0/0"],
            description="FastAPI",
        ),
        aws.ec2.SecurityGroupIngressArgs(
            protocol="tcp",
            from_port=3000,
            to_port=3000,
            cidr_blocks=["0.0.0.0/0"],
            description="Grafana",
        ),
        aws.ec2.SecurityGroupIngressArgs(
            protocol="tcp",
            from_port=5000,
            to_port=5000,
            cidr_blocks=["0.0.0.0/0"],
            description="MLflow",
        ),
        aws.ec2.SecurityGroupIngressArgs(
            protocol="tcp",
            from_port=9090,
            to_port=9090,
            cidr_blocks=["0.0.0.0/0"],
            description="Prometheus",
        ),
    ],
    egress=[
        aws.ec2.SecurityGroupEgressArgs(
            protocol="-1",
            from_port=0,
            to_port=0,
            cidr_blocks=["0.0.0.0/0"],
            description="All outbound",
        )
    ],
    tags={**COMMON_TAGS, "Name": f"{PREFIX}-sg"},
)

iam_role = aws.iam.Role(
    f"{PREFIX}-instance-role",
    assume_role_policy="""{
        "Version": "2012-10-17",
        "Statement": [{
            "Effect": "Allow",
            "Principal": {"Service": "ec2.amazonaws.com"},
            "Action": "sts:AssumeRole"
        }]
    }""",
    tags=COMMON_TAGS,
)

aws.iam.RolePolicyAttachment(
    f"{PREFIX}-s3-readwrite",
    role=iam_role.name,
    policy_arn="arn:aws:iam::aws:policy/AmazonS3FullAccess",
)

aws.iam.RolePolicyAttachment(
    f"{PREFIX}-ecr-pull",
    role=iam_role.name,
    policy_arn="arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryReadOnly",
)

instance_profile = aws.iam.InstanceProfile(
    f"{PREFIX}-instance-profile",
    role=iam_role.name,
)

ec2 = aws.ec2.Instance(
    f"{PREFIX}-ec2",
    ami=ami.id,
    instance_type="t2.micro",
    subnet_id=subnet.id,
    vpc_security_group_ids=[sg.id],
    iam_instance_profile=instance_profile.name,
    key_name=key_pair.key_name if key_pair else None,
    user_data="""#!/bin/bash
yum update -y
yum install -y docker git awscli
usermod -aG docker ec2-user
systemctl start docker
systemctl enable docker
curl -SL https://github.com/docker/compose/releases/download/v2.27.1/docker-compose-linux-x86_64 -o /usr/local/bin/docker-compose
chmod +x /usr/local/bin/docker-compose
ln -sf /usr/local/bin/docker-compose /usr/bin/docker-compose
mkdir -p /opt/modelserve
chown -R ec2-user:ec2-user /opt/modelserve
""",
    tags={**COMMON_TAGS, "Name": f"{PREFIX}-ec2"},
)

ecr = aws.ecr.Repository(
    f"{PREFIX}-ecr",
    force_delete=True,
    tags=COMMON_TAGS,
)

s3_bucket = aws.s3.Bucket(
    f"{PREFIX}-mlflow",
    bucket_prefix=f"{PREFIX}-mlflow-artifacts-",
    force_destroy=True,
    tags={**COMMON_TAGS, "Name": f"{PREFIX}-mlflow"},
)

pulumi.export("vpc_id", vpc.id)
pulumi.export("subnet_id", subnet.id)
pulumi.export("security_group_id", sg.id)
pulumi.export("ec2_public_ip", ec2.public_ip)
pulumi.export("ecr_repo_url", ecr.repository_url)
pulumi.export("s3_bucket_name", s3_bucket.id)
